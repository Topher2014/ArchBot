"""
Search API routes.
"""

from flask import Blueprint, request, jsonify, current_app
from rdb.retrieval.retriever import DocumentRetriever
from rdb.storage.database import DatabaseManager
from rdb.utils.helpers import Timer
import logging

search_bp = Blueprint('search', __name__)
logger = logging.getLogger(__name__)


@search_bp.route('/query', methods=['POST'])
def search_query():
    """Search documents."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        top_k = data.get('top_k', 5)
        refine_query = data.get('refine_query', False)
        
        config = current_app.config['RDB_CONFIG']
        
        # Initialize retriever
        retriever = DocumentRetriever(config)
        if not retriever.load_index():
            return jsonify({'error': 'Search index not available. Please build index first.'}), 503
        
        # Log search
        db_manager = DatabaseManager(config)
        search_data = {
            'original_query': query,
            'top_k': top_k,
            'query_refinement_enabled': refine_query
        }
        
        # Perform search
        with Timer() as timer:
            results = retriever.search(query, top_k=top_k, refine_query=refine_query)
        
        # Update search log
        search_data.update({
            'refined_query': results[0]['final_query'] if results else query,
            'results_count': len(results),
            'search_time_ms': int(timer.elapsed * 1000)
        })
        db_manager.log_search(search_data)
        
        return jsonify({
            'query': query,
            'refined_query': results[0]['final_query'] if results else query,
            'results': results,
            'search_time_ms': search_data['search_time_ms'],
            'total_results': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/history', methods=['GET'])
def search_history():
    """Get search history."""
    try:
        limit = request.args.get('limit', 20, type=int)
        config = current_app.config['RDB_CONFIG']
        
        db_manager = DatabaseManager(config)
        history = db_manager.get_recent_searches(limit)
        stats = db_manager.get_search_stats()
        
        return jsonify({
            'history': history,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Search history error: {e}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/suggestions', methods=['GET'])
def search_suggestions():
    """Get search suggestions (simple implementation)."""
    try:
        config = current_app.config['RDB_CONFIG']
        
        # Simple suggestions based on common Arch Wiki topics
        suggestions = [
            "wifi connection problems",
            "install arch linux",
            "graphics driver issues", 
            "sound not working",
            "pacman package manager",
            "systemd services",
            "bluetooth configuration",
            "desktop environment setup",
            "network configuration",
            "boot loader grub"
        ]
        
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return jsonify({'error': str(e)}), 500
