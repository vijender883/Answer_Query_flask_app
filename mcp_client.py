# mcp_client.py
import requests
import time
from typing import List, Dict, Any, Optional

class MCPClient:
    """Client for communicating with the MCP server to discover and execute tools"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.tools_cache = []
        self.tools_cache_timestamp = 0
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    def _make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """Make request to MCP server"""
        url = f"{self.server_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
            
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"MCP request error: {e}")
            return {"error": str(e)}
    
    def discover_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Discover available tools from MCP server with caching"""
        current_time = time.time()
        
        # Use cached tools if available and not expired
        if not force_refresh and self.tools_cache and (current_time - self.tools_cache_timestamp) < self.cache_ttl:
            return self.tools_cache
            
        # Fetch tools from MCP server
        tools_response = self._make_request("/api/tools")
        
        if "error" in tools_response:
            # On error, return empty list if no cache, or expired cache otherwise
            return [] if not self.tools_cache else self.tools_cache
            
        # Update cache
        self.tools_cache = tools_response.get("tools", [])
        self.tools_cache_timestamp = current_time
        
        return self.tools_cache
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool details by name"""
        tools = self.discover_tools()
        
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
                
        return None
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool on the MCP server"""
        if params is None:
            params = {}
            
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
            
        endpoint = f"/api/tools/{tool_name}/execute"
        result = self._make_request(endpoint, method="POST", data=params)
        
        return result