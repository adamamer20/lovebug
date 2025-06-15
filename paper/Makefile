# Makefile for Lovebug paper
# Following Adam's coding standards for DRY, automated workflows

.PHONY: all clean paper view structure help

# Build the paper
paper: figures
	@echo "Building paper with Quarto..."
	quarto render paper.qmd
	chmod +x scripts/move_build_files.sh
	 ./scripts/move_build_files.sh
	@echo "Paper built successfully! Output: _build/paper.pdf"

# View the paper (opens in host browser)
view: paper
	@echo "Opening paper in browser..."
	"$$BROWSER" _build/paper.pdf || echo "Set BROWSER environment variable to view PDF"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf _build/
	rm -f paper.pdf
	@echo "Clean complete!"

# Check Quarto is available
check:
	@echo "Checking Quarto installation..."
	@which quarto > /dev/null || (echo "Quarto not found! Install from https://quarto.org/docs/get-started/" && exit 1)
	@echo "Quarto is available: $$(quarto --version)"

# Help target
help:
	@echo "Available targets:"
	@echo "  paper     - Build the paper PDF"
	@echo "  view      - Open paper in browser"
	@echo "  clean     - Remove generated files"
	@echo "  check     - Check Quarto installation"
	@echo "  help      - Show this help message"
