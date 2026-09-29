.PHONY: check

check: ## Format check, Clippy with warnings denied, and tests.
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	cargo test
