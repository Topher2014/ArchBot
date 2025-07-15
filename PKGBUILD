# Maintainer: Topher Ludlow <topherludlow@protonmail.com>
pkgname=rdb
pkgver=0.1.0
pkgrel=1
pkgdesc="Retrieval Database for Arch Wiki documentation with semantic search"
arch=('any')
url="https://github.com/Topher2014/rdb"
license=('MIT')
depends=(
    'python'
    'python-requests'
    'python-beautifulsoup4'
    'python-lxml'
    'python-numpy'
    'python-pandas'
    'python-pytorch'
    'python-tqdm'
    'python-dotenv'
    'python-click'
    'python-rich'
    'python-flask'
    'python-pip'
)
makedepends=(
    'python-build'
    'python-installer'
    'python-wheel'
    'python-setuptools'
)
optdepends=(
    'python-pytorch-cuda: CUDA support for PyTorch'
    'cuda: NVIDIA GPU support'
)
checkdepends=(
    'python-pytest'
    'python-pytest-cov'
)
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz"
        "$pkgname-data-$pkgver.tar.gz::$url/releases/download/v$pkgver/rdb-data-v$pkgver.tar.gz")
sha256sums=('SKIP'  # Update with actual checksum of source code
            'SKIP') # Update with actual checksum of data archive

build() {
    cd "$pkgname-$pkgver"
    python -m build --wheel --no-isolation
}

check() {
    cd "$pkgname-$pkgver"
    # Run tests if they don't require external resources
    # python -m pytest tests/ || warning "Tests failed"
}

package() {
    cd "$pkgname-$pkgver"
    python -m installer --destdir="$pkgdir" dist/*.whl
    
    # Install license and documentation
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
    
    # Install pre-built data
    install -dm755 "$pkgdir/usr/share/$pkgname"
    cd "$srcdir"
    if [[ -f "$pkgname-data-$pkgver.tar.gz" ]]; then
        bsdtar -xf "$pkgname-data-$pkgver.tar.gz"
        cp -r data/* "$pkgdir/usr/share/$pkgname/"
    fi
    
    # Install missing Python dependencies via pip
    # This installs them into the system Python site-packages
    python -m pip install --root="$pkgdir" --no-deps \
        transformers \
        sentence-transformers \
        faiss-cpu \
        accelerate
    
    # Install GPU upgrade helper script
    install -Dm755 /dev/stdin "$pkgdir/usr/bin/rdb-enable-gpu" << 'EOF'
#!/bin/bash
echo "Upgrading RDB to GPU acceleration..."
echo "This will install faiss-gpu and may take a few minutes..."
echo ""

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Install faiss-gpu
if pip install --user faiss-gpu; then
    echo ""
    echo "✓ GPU acceleration enabled!"
    echo ""
    echo "To use GPU acceleration:"
    echo "  export RDB_USE_GPU=true"
    echo "  rdb search 'your query'"
    echo ""
    echo "Or add to your shell profile (~/.bashrc, ~/.zshrc):"
    echo "  echo 'export RDB_USE_GPU=true' >> ~/.bashrc"
    echo ""
    echo "Test GPU availability:"
    echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
else
    echo ""
    echo "✗ Failed to install faiss-gpu"
    echo "You may need to install CUDA toolkit and drivers first."
    echo "See: https://wiki.archlinux.org/title/GPGPU#CUDA"
fi
EOF
}

post_install() {
    echo "RDB has been installed with all dependencies and pre-built data!"
    echo ""
    echo "Start searching immediately:"
    echo "  rdb search 'wifi configuration'"
    echo "  rdb search --interactive"
    echo ""
    echo "For web interface:"
    echo "  rdb web"
    echo ""
    echo "For GPU acceleration (if you have NVIDIA GPU):"
    echo "  rdb-enable-gpu"
    echo ""
    echo "Data is stored in ~/.local/share/rdb"
    echo "Set RDB_DATA_DIR environment variable to use a different location"
}

post_upgrade() {
    echo "RDB has been upgraded!"
    echo ""
    echo "Your existing data in ~/.local/share/rdb is preserved."
    echo ""
    echo "If you were using GPU acceleration, you may need to reinstall faiss-gpu:"
    echo "  rdb-enable-gpu"
    echo ""
    echo "To rebuild with fresh data:"
    echo "  rdb scrape && rdb build"
}

post_remove() {
    echo "RDB has been removed."
    echo ""
    echo "Your data directory ~/.local/share/rdb has been preserved."
    echo "Remove it manually if you don't need it:"
    echo "  rm -rf ~/.local/share/rdb"
    echo ""
    echo "Python dependencies installed by RDB remain in the system."
    echo "Remove them manually if desired:"
    echo "  pip uninstall transformers sentence-transformers faiss-cpu accelerate"
}
