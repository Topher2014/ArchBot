intainer: Topher Ludlow <topherludlow@protonmail.com>
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
    'python-transformers'
    'python-sentence-transformers'
    'python-faiss'
    'python-tqdm'
    'python-dotenv'
    'python-click'
    'python-rich'
    'python-flask'
)
makedepends=(
    'python-build'
    'python-installer'
    'python-wheel'
    'python-setuptools'
)
optdepends=(
    'python-faiss-gpu: GPU acceleration for embeddings'
    'python-accelerate: Hardware acceleration support'
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
    
    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    
    # Install documentation  
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
    
    # Install pre-built data to system location
    install -dm755 "$pkgdir/usr/share/$pkgname"
    
    # Extract and install the data archive
    cd "$srcdir"
    if [[ -f "$pkgname-data-$pkgver.tar.gz" ]]; then
        bsdtar -xf "$pkgname-data-$pkgver.tar.gz"
        cp -r data/* "$pkgdir/usr/share/$pkgname/"
    fi
}

post_install() {
    echo "RDB has been installed with pre-built Arch Wiki data!"
    echo ""
    echo "The data will be automatically set up on first use."
    echo "Start searching immediately:"
    echo "  rdb search 'wifi configuration'"
    echo "  rdb search --interactive"
    echo ""
    echo "For web interface:"
    echo "  rdb web"
    echo ""
    echo "Data will be stored in ~/.local/share/rdb"
}

post_upgrade() {
    echo "RDB has been upgraded!"
    echo ""
    echo "Your existing data in ~/.local/share/rdb is preserved."
    echo "To use the new pre-built data:"
    echo "  rm -rf ~/.local/share/rdb"
    echo "  /usr/share/rdb/setup-data.sh"
    echo ""
    echo "Or to update with fresh data:"
    echo "  rdb scrape && rdb build"
}
