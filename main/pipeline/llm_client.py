"""LLM client abstraction: one place that knows endpoints, JSON modes, budgets.

Three kinds:
  ollama  — Ollama native /api/chat with format=<json schema>
  openai  — any OpenAI-compatible endpoint (vLLM, etc.) with json_schema
  mock    — deterministic fixture client (no network); answers from ground truth
            so the GRAPH can be exercised on machines without the model.

All calls return (parsed_dict_or_text, usage_dict). Token usage is accumulated
on the client for budget reporting.
"""

import json
import os
import re
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(profile: str | None = None) -> dict:
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    name = profile or os.environ.get('PIPELINE_PROFILE') or cfg['active_profile']
    if name not in cfg['profiles']:
        raise KeyError(f'unknown profile {name!r}')
    out = dict(cfg['profiles'][name])
    out['profile_name'] = name
    out['policy'] = cfg['policy']
    return out


class LLMClient:
    def __init__(self, profile: str | None = None):
        self.cfg = load_config(profile)
        self.kind = self.cfg['kind']
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.trace_path = None   # set per-problem by run_pipeline -> traces.jsonl
        if self.kind == 'openai':
            from openai import OpenAI
            key = os.environ.get(self.cfg.get('api_key_env', ''), 'none') or 'none'
            self._oai = OpenAI(base_url=self.cfg['base_url'], api_key=key,
                               timeout=self.cfg.get('timeout_s', 120))
        elif self.kind == 'ollama':
            import httpx
            self._http = httpx.Client(base_url=self.cfg['base_url'],
                                      timeout=self.cfg.get('timeout_s', 180))
        elif self.kind == 'mock':
            self._mock = MockBackend()
        else:
            raise ValueError(f'unknown client kind {self.kind!r}')

    # ── core call ────────────────────────────────────────────────────
    def complete_json(self, node: str, system: str, user: str,
                      schema: dict, meta: dict | None = None) -> dict:
        """One JSON-constrained completion. Raises on unparseable output.

        Every call (including failed parses) is appended to trace_path when
        set — full system/user/raw text, for diagnosis and tracing.
        """
        self.calls += 1
        meta = meta or {}
        effort = (self.cfg.get('reasoning_effort') or {}).get(node)
        if effort and self.kind in ('ollama', 'openai'):
            system = f'Reasoning: {effort}\n\n{system}'

        t0 = time.perf_counter()
        raw, err = None, None
        tok_in = tok_out = 0
        try:
            if self.kind == 'mock':
                result = self._mock.answer(node, user, schema, meta)
                raw = json.dumps(result)
            elif self.kind == 'ollama':
                r = self._http.post('/api/chat', json={
                    'model': self.cfg['model'],
                    'messages': [{'role': 'system', 'content': system},
                                 {'role': 'user', 'content': user}],
                    'format': schema,          # Ollama structured outputs
                    'stream': False,
                })
                r.raise_for_status()
                body = r.json()
                tok_in = body.get('prompt_eval_count', 0)
                tok_out = body.get('eval_count', 0)
                self.tokens_in += tok_in
                self.tokens_out += tok_out
                raw = body['message']['content']
                result = json.loads(raw)
            else:   # openai-compatible
                kwargs = {}
                if self.cfg.get('json_mode') == 'openai_schema':
                    kwargs['response_format'] = {
                        'type': 'json_schema',
                        'json_schema': {'name': f'{node}_out', 'schema': schema,
                                        'strict': True}}
                if 'temperature' in self.cfg:
                    kwargs['temperature'] = self.cfg['temperature']
                if 'max_tokens' in self.cfg:
                    kwargs['max_tokens'] = self.cfg['max_tokens']
                resp = self._oai.chat.completions.create(
                    model=self.cfg['model'],
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': user}],
                    **kwargs)
                if resp.usage:
                    tok_in = resp.usage.prompt_tokens or 0
                    tok_out = resp.usage.completion_tokens or 0
                    self.tokens_in += tok_in
                    self.tokens_out += tok_out
                raw = resp.choices[0].message.content
                result = _parse_json_loose(raw)
            return result
        except Exception as e:
            err = str(e)
            raise
        finally:
            if self.trace_path:
                rec = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'node': node, 'model': self.cfg.get('model', self.kind),
                       'problem_id': meta.get('problem_id'),
                       'system': system, 'user': user, 'raw': raw, 'error': err,
                       'tokens_in': tok_in, 'tokens_out': tok_out,
                       'latency_ms': round(1000 * (time.perf_counter() - t0))}
                try:
                    with open(self.trace_path, 'a') as f:
                        f.write(json.dumps(rec, default=str) + '\n')
                except OSError:
                    pass   # tracing must never break a run

    def usage(self) -> dict:
        return {'calls': self.calls, 'tokens_in': self.tokens_in,
                'tokens_out': self.tokens_out}


def _parse_json_loose(text: str) -> dict:
    """Parse JSON possibly wrapped in prose/code fences (plain-mode floor)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r'\{.*\}', text or '', re.DOTALL)
        if not m:
            raise ValueError(f'no JSON object in model output: {text!r:.200}')
        return json.loads(m.group(0))


class MockBackend:
    """Deterministic fixture answers — exercises graph mechanics, NOT model skill.

    Reads eval/reference/ground_truth_fills.yaml; keyed by problem_id passed in
    meta. For unknown problems it answers with template defaults (still valid),
    so graph paths remain testable. Clearly labeled: results from this backend
    are PLUMBING tests only.
    """

    def __init__(self):
        gt_path = ROOT / 'eval/reference/ground_truth_fills.yaml'
        self.gt = yaml.safe_load(gt_path.read_text()) if gt_path.exists() else {}

    def answer(self, node: str, user: str, schema: dict, meta: dict) -> dict:
        pid = meta.get('problem_id', '')
        rec = (self.gt.get('problems') or {}).get(pid, {})
        if node == 'selector':
            return {'template_id': rec.get('template_id', 'gc_polyc_pb2d')}
        if node == 'slot_filler':
            group_props = list((schema.get('properties') or {}))
            fills = rec.get('fills', {})
            out = {}
            for k in group_props:
                if k in fills:
                    out[k] = fills[k]
            return out
        if node == 'cross_checker':
            return {'ok': True, 'reason': 'mock backend: fills match ground truth by construction'}
        raise ValueError(f'mock backend: unknown node {node!r}')
