#!/usr/bin/env python3
import argparse
import trimesh

def main():
    parser = argparse.ArgumentParser(
        description="Visualiza un archivo .glb usando trimesh"
    )
    parser.add_argument("glb", help="Ruta al archivo .glb")
    args = parser.parse_args()

    # Carga la escena o malla
    scene = trimesh.load(args.glb)
    # Abre la ventana de visualización
    scene.show()

if __name__ == "__main__":
    main()
