import argparse
import sys
import os
import cv2
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import map_coordinates
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def convert_galactic_to_equatorial(input_path, output_path, width):
    """
    Core conversion logic with Rich status updates.
    """
    
    # --- 1. Load Image ---
    with console.status(f"[bold green]Loading {input_path}...", spinner="dots"):
        img_galactic = cv2.imread(input_path, -1)
        if img_galactic is None:
            raise FileNotFoundError(f"Could not load file at {input_path}")

    h_in, w_in = img_galactic.shape[:2]
    height = width // 2
    
    console.print(f"[cyan]Input Resolution:[/cyan] {w_in}x{h_in}")
    console.print(f"[cyan]Target Resolution:[/cyan] {width}x{height}")

    # --- 2. Generate Coordinate Grid ---
    with console.status("[bold blue]Generating coordinate grids (Astropy)...", spinner="earth"):
        # Create output grid (Equatorial)
        x_indices = np.linspace(0, width - 1, width)
        y_indices = np.linspace(0, height - 1, height)
        x_grid, y_grid = np.meshgrid(x_indices, y_indices)

        # Convert output pixels to RA/Dec
        # Assuming standard map: Right edge = 0 deg RA, Left edge = 360 deg RA
        ra_deg = (x_grid / width) * 360.0
        dec_deg = ((y_grid / height) * 180.0) - 90.0

        # Convert RA/Dec to Galactic (l, b)
        coords_eq = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        coords_gal = coords_eq.galactic
        
        l = coords_gal.l.deg
        b = coords_gal.b.deg

        # Map Galactic (l, b) to Input Pixel Coordinates
        # Assumes Input Image is 0..360 deg longitude
        x_src = (l / 360.0) * w_in
        x_src = np.mod(x_src, w_in)
        
        y_src = ((b + 90) / 180.0) * h_in
        y_src = np.clip(y_src, 0, h_in - 1)

    # --- 3. Remap Channels ---
    # We use a Progress bar here since we process channels iteratively
    output_channels = []
    
    if len(img_galactic.shape) == 2:
        channels = [img_galactic]
    else:
        channels = cv2.split(img_galactic)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("[green]Remapping pixels...", total=len(channels))
        
        for i, chan in enumerate(channels):
            # Order=1 (Linear) is fast. Change to order=3 (Cubic) for higher quality.
            remapped_chan = map_coordinates(chan, [y_src, x_src], order=1, mode='wrap')
            output_channels.append(remapped_chan)
            progress.advance(task)

    # --- 4. Merge and Save ---
    with console.status(f"[bold yellow]Saving to {output_path}...", spinner="dots"):
        if len(output_channels) > 1:
            img_equatorial = cv2.merge(output_channels)
        else:
            img_equatorial = output_channels[0]
            
        cv2.imwrite(output_path, img_equatorial)

    console.print(Panel(f"[bold green]Success![/bold green]\nImage saved to [underline]{output_path}[/underline]", title="Done"))

def main():
    parser = argparse.ArgumentParser(description="Convert Galactic Sky Map to Equatorial (J2000)")
    parser.add_argument("-i", "--input", help="Path to input image (Galactic)")
    parser.add_argument("-o", "--output", help="Path to output image (Equatorial)")
    parser.add_argument("-w", "--width", type=int, default=4096, help="Output width (default: 4096)")
    
    args = parser.parse_args()

    console.print(Panel("[bold magenta]Galactic -> Equatorial Converter[/bold magenta]", expand=False))

    # Input Handling: Check args, otherwise ask interactively
    input_path = args.input
    if not input_path:
        input_path = Prompt.ask("[bold cyan]Enter input file path[/bold cyan]")
        
    if not os.path.exists(input_path):
        console.print(f"[bold red]Error:[/bold red] File '{input_path}' not found.")
        sys.exit(1)

    # Output Handling
    output_path = args.output
    if not output_path:
        # Generate a smart default based on input name
        base, ext = os.path.splitext(input_path)
        default_out = f"{base}_equatorial{ext}"
        output_path = Prompt.ask("[bold cyan]Enter output file path[/bold cyan]", default=default_out)

    try:
        convert_galactic_to_equatorial(input_path, output_path, args.width)
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()