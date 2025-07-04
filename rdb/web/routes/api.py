"""
General API routes.
"""

from flask import Blueprint, jsonify, current_app
from rdb import __version__, __author__
import logging

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': __version__,
        'author': __author__
    })


@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    try:
        config = current_app.config['RDB_CONFIG']
        
        return jsonify({
            'data_dir': str(config.data_dir),
            'embedding_model': config.embedding_model,
            'device': config.device,
            'default_top_k': config.default_top_k,
            'enable_query_refinement': config.enable_query_refinement,
            'chunk_sizes': {
                'small': config.chunk_size_small,
                'medium': config.chunk_size_medium,
                'large': config.chunk_size_large
            }
        })
        
    except Exception as e:
        logger.error(f"Get config error: {e}")
        return jsonify({'error': str(e)}), 500
