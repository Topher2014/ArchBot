"""
Flask application for RDB web interface.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rdb.config.settings import Config
from rdb.utils.logging import setup_logging
from .routes import search_bp, api_bp


def create_app(data_dir=None, debug=False):
    """Create Flask application."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    app.config['DEBUG'] = debug
    
    # Initialize RDB config
    config = Config(data_dir=data_dir)
    app.config['RDB_CONFIG'] = config
    
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level=log_level)
    
    # Register blueprints (removed admin_bp)
    app.register_blueprint(search_bp, url_prefix='/api/search')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Main routes
    @app.route('/')
    def index():
        """Main page."""
        return render_template('index.html')
    
    @app.route('/search')
    def search_page():
        """Search page."""
        return render_template('search.html')
    
    @app.errorhandler(404)
    def not_found(error):
        """404 error handler."""
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500 error handler."""
        return jsonify({'error': 'Internal server error'}), 500
    
    return app


if __name__ == '__main__':
    app = create_app(debug=True)
    app.run(host='0.0.0.0', port=5000)
