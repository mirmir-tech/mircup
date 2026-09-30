.PHONY: check

check: ## Format check, Clippy on macOS and both Linux targets, and tests.
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	cargo test
	cargo clippy --all-targets --target x86_64-unknown-linux-gnu -- -D warnings
	cargo clippy --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
