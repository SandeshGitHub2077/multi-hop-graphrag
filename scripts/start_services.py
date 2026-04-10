#!/usr/bin/env python3
"""Auto-start services for the GraphRAG pipeline.

Detects and starts Neo4j and Ollama if not already running.
"""

import subprocess
import time
import socket
import argparse


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


# Use 127.0.0.1 instead of localhost for consistency
NEO4J_HOST = "127.0.0.1"
OLLAMA_HOST = "127.0.0.1"


def check_neo4j() -> bool:
    """Check if Neo4j is running and responsive."""
    # Check if bolt port is open (more reliable than process name)
    if is_port_open("127.0.0.1", 7687):
        return True
    return False


def check_ollama() -> bool:
    """Check if Ollama is running and responsive."""
    # Check if API port is open (more reliable than process name)
    if is_port_open("127.0.0.1", 11434):
        return True
    return False


def start_neo4j() -> bool:
    """Start Neo4j and wait for it to be ready."""
    print("Starting Neo4j...")
    try:
        subprocess.run(["neo4j", "start"], check=True, timeout=30)
        
        # Wait for Neo4j to be ready (up to 60 seconds)
        for _ in range(60):
            if is_port_open("localhost", 7687, timeout=1.0):
                # Give it a bit more time to be fully ready
                time.sleep(2)
                print("Neo4j started successfully")
                return True
            time.sleep(1)
        
        print("Warning: Neo4j may not be fully ready")
        return True
    except FileNotFoundError:
        print("Error: neo4j command not found. Is it installed?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error starting Neo4j: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def start_ollama() -> bool:
    """Start Ollama and wait for it to be ready."""
    print("Starting Ollama...")
    try:
        subprocess.Popen(
            ["/usr/local/bin/ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for Ollama to be ready (up to 30 seconds)
        for _ in range(30):
            if is_port_open("127.0.0.1", 11434, timeout=1.0):
                time.sleep(1)
                print("Ollama started successfully")
                return True
            time.sleep(1)
        
        print("Warning: Ollama may not be fully ready")
        return True
    except FileNotFoundError:
        print("Error: ollama command not found. Is it installed?")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def ensure_neo4j(require: bool = True) -> bool:
    """Ensure Neo4j is running."""
    if check_neo4j():
        print("Neo4j already running")
        return True
    
    if not require:
        print("Neo4j not running (skipping)")
        return True
    
    return start_neo4j()


def ensure_ollama(require: bool = True) -> bool:
    """Ensure Ollama is running."""
    if check_ollama():
        print("Ollama already running")
        return True
    
    if not require:
        print("Ollama not running (skipping)")
        return True
    
    return start_ollama()


def main():
    parser = argparse.ArgumentParser(description="Auto-start GraphRAG services")
    parser.add_argument("--neo4j", action="store_true", default=True, help="Start Neo4j (default: true)")
    parser.add_argument("--no-neo4j", action="store_true", help="Skip Neo4j")
    parser.add_argument("--ollama", action="store_true", default=True, help="Start Ollama (default: true)")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama")
    parser.add_argument("--wait", type=int, default=30, help="Max wait time for each service")
    args = parser.parse_args()
    
    print("=" * 50)
    print("GraphRAG Service Starter")
    print("=" * 50)
    
    success = True
    
    if args.neo4j and not args.no_neo4j:
        if not ensure_neo4j(require=True):
            success = False
    
    if args.ollama and not args.no_ollama:
        if not ensure_ollama(require=True):
            success = False
    
    print("=" * 50)
    if success:
        print("All services ready!")
    else:
        print("Warning: Some services may not be available")


if __name__ == "__main__":
    main()