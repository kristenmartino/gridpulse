"""Tests for observability.ip_in_allowlist — exact-IP + CIDR matching (#253).

The CIDR form exists so a residential IPv6 stays allowlisted through host-bit
rotation: the /64 prefix is stable even as the interface identifier cycles.
"""

from __future__ import annotations

from observability import ip_in_allowlist


class TestExactMatch:
    def test_ipv4_exact(self):
        assert ip_in_allowlist("1.2.3.4", ["1.2.3.4"]) is True
        assert ip_in_allowlist("1.2.3.5", ["1.2.3.4"]) is False

    def test_ipv6_exact_formatting_equivalent(self):
        # "::1" and its fully-expanded form are the same address.
        assert ip_in_allowlist("::1", ["0:0:0:0:0:0:0:1"]) is True

    def test_matches_any_entry_in_list(self):
        assert ip_in_allowlist("9.9.9.9", ["1.1.1.1", "9.9.9.9", "8.8.8.8"]) is True


class TestCidrMatch:
    def test_ipv4_cidr(self):
        assert ip_in_allowlist("10.0.5.7", ["10.0.0.0/8"]) is True
        assert ip_in_allowlist("11.0.5.7", ["10.0.0.0/8"]) is False

    def test_ipv6_64_prefix_survives_host_rotation(self):
        """The whole point (#253): two different host identifiers in the same
        /64 both match, so a rotating residential IPv6 stays allowlisted."""
        cidr = ["2600:1700:f890:2740::/64"]
        assert ip_in_allowlist("2600:1700:f890:2740:5cbf:75b1:f3d6:464f", cidr) is True
        assert ip_in_allowlist("2600:1700:f890:2740:aaaa:bbbb:cccc:dddd", cidr) is True
        # A different /64 (…:2741) is outside the prefix.
        assert ip_in_allowlist("2600:1700:f890:2741::1", cidr) is False

    def test_mixed_exact_and_cidr_entries(self):
        entries = ["108.233.134.69", "2600:1700:f890:2740::/64"]
        assert ip_in_allowlist("108.233.134.69", entries) is True  # exact IPv4
        assert ip_in_allowlist("2600:1700:f890:2740:1:2:3:4", entries) is True  # in /64
        assert ip_in_allowlist("203.0.113.9", entries) is False


class TestFailClosed:
    def test_empty_ip_never_matches(self):
        assert ip_in_allowlist("", ["1.2.3.4", "0.0.0.0/0"]) is False

    def test_malformed_ip_never_matches(self):
        assert ip_in_allowlist("not-an-ip", ["1.2.3.4"]) is False

    def test_malformed_entry_is_skipped_not_fatal(self):
        # A bad entry must not crash the check nor match; a good later entry still works.
        assert ip_in_allowlist("1.2.3.4", ["garbage/xx", "1.2.3.4"]) is True
        assert ip_in_allowlist("1.2.3.4", ["garbage/xx"]) is False

    def test_empty_and_whitespace_entries_skipped(self):
        assert ip_in_allowlist("1.2.3.4", ["", "  ", "1.2.3.4"]) is True
        assert ip_in_allowlist("1.2.3.4", []) is False
