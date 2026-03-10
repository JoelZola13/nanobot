"""Multi-agent routing — maps inbound messages to target agents."""

from nanobot.routing.models import BindingRule, ResolvedBinding
from nanobot.routing.resolver import BindingResolver

__all__ = ["BindingRule", "ResolvedBinding", "BindingResolver"]
