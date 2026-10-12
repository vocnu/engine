import pytest

from vocnu_engine.costs import BudgetExceeded, Receipt


def test_openai_cost_uses_cached_and_fresh_input_rates():
    receipt = Receipt(max_usd=1.0)
    usd = receipt.openai('gpt-5.6-terra', 'flex', {'input_tokens': 10_000, 'output_tokens': 2_000,
                                                   'input_tokens_details': {'cached_tokens': 4_000}})
    assert usd == pytest.approx((6_000 * 1.00 + 4_000 * 0.10 + 2_000 * 6.00) / 1e6)
    assert receipt.entries[0]['cached_tokens'] == 4_000


def test_budget_check_refuses_overspend():
    receipt = Receipt(max_usd=0.10)
    receipt.add('speech', 'elevenlabs', 0.08, characters=440)
    receipt.check(0.02, 'more speech')
    with pytest.raises(BudgetExceeded, match='more speech'):
        receipt.check(0.03, 'more speech')
    assert receipt.as_dict()['total_usd'] == 0.08
