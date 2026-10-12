"""Provider cost receipt for one job, with a hard budget that stops spending before it is exceeded."""
from dataclasses import dataclass, field

# USD per 1M tokens: input, cached input, cache write, output. OpenAI pricing page, checked 2026-10-07.
OPENAI_RATES = {
    ('gpt-6.1-sol', 'default'): (2.00, 0.10, 2.50, 10.00),
    ('gpt-6.1-sol', 'flex'): (1.00, 0.05, 1.25, 5.00),
    ('gpt-5.6-terra', 'default'): (2.00, 0.20, 2.50, 12.00),
    ('gpt-5.6-terra', 'flex'): (1.00, 0.10, 1.25, 6.00),
    ('gpt-5.6-luna', 'default'): (0.20, 0.02, 0.25, 1.20),
    ('gpt-5.6-luna', 'flex'): (0.10, 0.01, 0.125, 0.60),
}
# ElevenLabs: $20 bought 110,000 credits (2026-09-23), the same rate as Creator ($22 / 121,000).
ELEVENLABS_USD_PER_CREDIT = 20 / 110_000
# Modal L4 GPU and CPU core list prices, checked 2026-10-07 (1 core = 2 vCPU); A10G is the fallback GPU.
MODAL_L4_USD_PER_SECOND = 0.000222
MODAL_A10G_USD_PER_SECOND = 0.000306
MODAL_GPU_USD_PER_SECOND = {'L4': MODAL_L4_USD_PER_SECOND, 'A10G': MODAL_A10G_USD_PER_SECOND}
MODAL_CORE_USD_PER_SECOND = 0.0000131


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Receipt:
    max_usd: float
    entries: list = field(default_factory=list)

    @property
    def total_usd(self):
        return round(sum(e['usd'] for e in self.entries), 6)

    def check(self, more_usd, what):
        """Refuse a paid step whose estimate would push the job over its budget."""
        if self.total_usd + more_usd > self.max_usd + 1e-9:
            raise BudgetExceeded(f'{what} would cost about ${more_usd:.4f}; job budget ${self.max_usd:.2f}, '
                                 f'already ${self.total_usd:.4f}')

    def add(self, kind, provider, usd, **units):
        self.entries.append({'kind': kind, 'provider': provider, 'usd': round(usd, 6), **units})

    def openai(self, model, tier, usage):
        # A model without a known price (a self-hosted engine's own choice) is recorded at $0.
        rate = OPENAI_RATES.get((model, tier), (0.0, 0.0, 0.0, 0.0))
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
        fresh = usage.get('input_tokens', 0) - cached
        usd = (fresh*rate[0] + cached*rate[1] + usage.get('output_tokens', 0)*rate[3]) / 1e6
        self.add('text', 'openai', usd, model=model, tier=tier, input_tokens=usage.get('input_tokens', 0),
                 cached_tokens=cached, output_tokens=usage.get('output_tokens', 0))
        return usd

    def as_dict(self):
        return {'currency': 'USD', 'total_usd': self.total_usd, 'max_usd': self.max_usd, 'entries': self.entries}
