"""Tests for Approval Policy module."""



from deepagents.approval import (
    DEFAULT_TOOL_CLASSIFICATIONS,
    ApprovalLedger,
    ApprovalPolicy,
    ApprovalRecord,
    ToolRiskLevel,
)


class TestToolRiskLevel:
    """Tests for ToolRiskLevel enum."""

    def test_ordering(self):
        """Test risk levels are ordered correctly."""
        assert ToolRiskLevel.SAFE < ToolRiskLevel.LOW
        assert ToolRiskLevel.LOW < ToolRiskLevel.MEDIUM
        assert ToolRiskLevel.MEDIUM < ToolRiskLevel.HIGH
        assert ToolRiskLevel.HIGH < ToolRiskLevel.CRITICAL


class TestApprovalPolicy:
    """Tests for ApprovalPolicy."""

    def test_default_classifications(self):
        """Test default tool classifications exist."""
        assert "read_file" in DEFAULT_TOOL_CLASSIFICATIONS
        assert "shell" in DEFAULT_TOOL_CLASSIFICATIONS
        assert DEFAULT_TOOL_CLASSIFICATIONS["read_file"] == ToolRiskLevel.SAFE
        assert DEFAULT_TOOL_CLASSIFICATIONS["shell"] == ToolRiskLevel.CRITICAL

    def test_get_risk_level_default(self):
        """Test getting risk level from defaults."""
        policy = ApprovalPolicy()
        assert policy.get_risk_level("read_file") == ToolRiskLevel.SAFE
        assert policy.get_risk_level("write_file") == ToolRiskLevel.MEDIUM
        assert policy.get_risk_level("shell") == ToolRiskLevel.CRITICAL

    def test_get_risk_level_override(self):
        """Test overriding default classifications."""
        policy = ApprovalPolicy(tool_classifications={"read_file": ToolRiskLevel.HIGH})
        assert policy.get_risk_level("read_file") == ToolRiskLevel.HIGH

    def test_requires_approval_default(self):
        """Test default approval requirements."""
        policy = ApprovalPolicy(min_risk_for_approval=ToolRiskLevel.MEDIUM)
        assert not policy.requires_approval("read_file")  # SAFE
        assert policy.requires_approval("write_file")  # MEDIUM
        assert policy.requires_approval("shell")  # CRITICAL

    def test_always_approve(self):
        """Test always_approve overrides risk level."""
        policy = ApprovalPolicy(
            min_risk_for_approval=ToolRiskLevel.CRITICAL,
            always_approve={"read_file"},
        )
        # Normally SAFE wouldn't need approval at CRITICAL threshold
        assert policy.requires_approval("read_file")

    def test_never_approve(self):
        """Test never_approve overrides risk level."""
        policy = ApprovalPolicy(
            min_risk_for_approval=ToolRiskLevel.SAFE,
            never_approve={"shell"},
        )
        # Normally CRITICAL would need approval
        assert not policy.requires_approval("shell")

    def test_to_interrupt_on_config(self):
        """Test generating interrupt_on config."""
        policy = ApprovalPolicy(min_risk_for_approval=ToolRiskLevel.HIGH)
        config = policy.to_interrupt_on_config()

        # HIGH and CRITICAL tools should be in config
        assert "shell" in config
        assert "web_search" in config
        # MEDIUM and below should not
        assert "write_file" not in config
        assert "read_file" not in config

    def test_get_tools_by_risk(self):
        """Test getting tools by risk level."""
        policy = ApprovalPolicy()
        critical_tools = policy.get_tools_by_risk(ToolRiskLevel.CRITICAL)
        assert "shell" in critical_tools
        assert "execute" in critical_tools


class TestApprovalLedger:
    """Tests for ApprovalLedger."""

    def test_record_and_query(self):
        """Test recording and querying decisions."""
        ledger = ApprovalLedger()

        record = ApprovalRecord(
            tool_name="shell",
            tool_args={"command": "ls"},
            decision="approve",
        )
        ledger.record(record)

        results = ledger.get_records(tool_name="shell")
        assert len(results) == 1
        assert results[0].decision == "approve"

    def test_filter_by_decision(self):
        """Test filtering by decision type."""
        ledger = ApprovalLedger()

        ledger.record(ApprovalRecord("shell", {}, "approve"))
        ledger.record(ApprovalRecord("shell", {}, "reject"))
        ledger.record(ApprovalRecord("write_file", {}, "approve"))

        approved = ledger.get_records(decision="approve")
        assert len(approved) == 2

        rejected = ledger.get_records(decision="reject")
        assert len(rejected) == 1

    def test_persistence(self, tmp_path):
        """Test ledger persistence to file."""
        ledger_path = tmp_path / "approvals.jsonl"

        # Create and write
        ledger1 = ApprovalLedger(ledger_path)
        ledger1.record(ApprovalRecord("shell", {"cmd": "test"}, "approve"))

        # Read back
        ledger2 = ApprovalLedger(ledger_path)
        records = ledger2.get_records()
        assert len(records) == 1
        assert records[0].tool_name == "shell"

    def test_stats(self):
        """Test statistics generation."""
        ledger = ApprovalLedger()

        ledger.record(ApprovalRecord("shell", {}, "approve"))
        ledger.record(ApprovalRecord("shell", {}, "approve"))
        ledger.record(ApprovalRecord("shell", {}, "reject"))
        ledger.record(ApprovalRecord("write_file", {}, "approve"))

        stats = ledger.get_stats()
        assert stats["total"] == 4
        assert stats["by_decision"]["approve"] == 3
        assert stats["by_decision"]["reject"] == 1
        assert stats["by_tool"]["shell"]["approve"] == 2
