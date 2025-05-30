# registry/plugins/cli_plugin.py
"""CLI plugin for command-line registry management."""

try:
    import click
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    click = typer = Console = Table = Panel = None

import sys
from typing import Type, Any, Dict, Optional, List
from .base import BaseRegistryPlugin

class CliPlugin(BaseRegistryPlugin):
    """Plugin for CLI operations."""
    
    @property
    def name(self) -> str:
        return "cli"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def dependencies(self) -> List[str]:
        return ["click", "typer", "rich"]
    
    def _on_initialize(self) -> None:
        if not CLI_AVAILABLE:
            raise ImportError("CLI dependencies required. Install with: pip install registry[cli-plugin]")
        
        self.console = Console()
        if self.registry_class:
            self._add_cli_methods()
    
    def _add_cli_methods(self) -> None:
        """Add CLI methods to the registry."""
        
        def print_registry_info() -> None:
            """Print registry information."""
            table = Table(title=f"{self.registry_class.__name__} Registry Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Registry Name", self.registry_class.__name__)
            table.add_row("Registry Type", self.registry_class.__class__.__name__)
            table.add_row("Total Artifacts", str(self.registry_class._len_mapping()))
            table.add_row("Strict Mode", str(getattr(self.registry_class, '_strict', False)))
            table.add_row("Abstract Mode", str(getattr(self.registry_class, '_abstract', False)))
            
            self.console.print(table)
        
        def print_artifacts(limit: Optional[int] = None) -> None:
            """Print all artifacts in the registry."""
            artifacts = dict(self.registry_class._get_mapping())
            
            if limit:
                artifacts = dict(list(artifacts.items())[:limit])
            
            table = Table(title=f"Artifacts in {self.registry_class.__name__}")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Type", style="yellow")
            
            for key, value in artifacts.items():
                table.add_row(
                    str(key)[:50] + ("..." if len(str(key)) > 50 else ""),
                    str(value)[:50] + ("..." if len(str(value)) > 50 else ""),
                    type(value).__name__
                )
            
            if limit and len(self.registry_class._get_mapping()) > limit:
                table.add_row("...", f"({len(self.registry_class._get_mapping()) - limit} more)", "")
            
            self.console.print(table)
        
        def interactive_register() -> None:
            """Interactive artifact registration."""
            self.console.print(Panel(f"Interactive Registration for {self.registry_class.__name__}", style="bold blue"))
            
            try:
                key = input("Enter key: ").strip()
                if not key:
                    self.console.print("[red]Key cannot be empty[/red]")
                    return
                
                value = input("Enter value: ").strip()
                if not value:
                    self.console.print("[red]Value cannot be empty[/red]")
                    return
                
                self.registry_class.register_artifact(key, value)
                self.console.print(f"[green]Successfully registered '{key}'[/green]")
                
            except Exception as e:
                self.console.print(f"[red]Registration failed: {e}[/red]")
        
        def create_cli_app() -> typer.Typer:
            """Create Typer CLI application."""
            app = typer.Typer(name=f"{self.registry_class.__name__.lower()}-registry")
            
            @app.command("info")
            def info():
                """Show registry information."""
                print_registry_info()
            
            @app.command("list")
            def list_artifacts_cmd(limit: Optional[int] = None):
                """List all artifacts."""
                print_artifacts(limit)
            
            @app.command("register")
            def register():
                """Interactively register an artifact."""
                interactive_register()
            
            @app.command("get")
            def get_artifact(key: str):
                """Get an artifact by key."""
                try:
                    value = self.registry_class.get_artifact(key)
                    self.console.print(f"[green]{key}[/green]: {value}")
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
            
            @app.command("remove")
            def remove_artifact(key: str):
                """Remove an artifact by key."""
                try:
                    self.registry_class.unregister_artifact(key)
                    self.console.print(f"[green]Removed '{key}'[/green]")
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
            
            @app.command("clear")
            def clear_registry():
                """Clear all artifacts."""
                confirm = input("Are you sure you want to clear all artifacts? (y/N): ")
                if confirm.lower() in ('y', 'yes'):
                    self.registry_class.clear_artifacts()
                    self.console.print("[green]Registry cleared[/green]")
                else:
                    self.console.print("[yellow]Operation cancelled[/yellow]")
            
            return app
        
        # Add methods to registry class as unbound functions
        setattr(self.registry_class, 'print_registry_info', print_registry_info)
        setattr(self.registry_class, 'print_artifacts', print_artifacts)
        setattr(self.registry_class, 'interactive_register', interactive_register)
        setattr(self.registry_class, 'create_cli_app', create_cli_app)

def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: registry-cli <registry_class_path>")
        sys.exit(1)
    
    # This would need to be implemented to dynamically load registry classes
    print("CLI plugin loaded. Use with specific registry classes.")

if __name__ == "__main__":
    main()
