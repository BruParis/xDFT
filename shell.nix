{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Specify the name of the development shell
  name = "python-dev-shell";

  # Define the build inputs (dependencies)
  buildInputs = [
    pkgs.python312  # Python 3.12 interpreter
    pkgs.python312Packages.pip  # pip for managing Python packages
    pkgs.python312Packages.numpy  # NumPy for numerical computing
    pkgs.gcc  # GCC libstdc++ and other libraries
  ];

  # Set environment variables if needed
  shellHook = ''
    # Optionally activate the virtual environment automatically
    if [ ! -d .venv ]; then
      python -m venv --system-site-packages .venv
    fi
    source .venv/bin/activate

    echo "Python development shell is ready."
    echo "Virtual environment located at $(pwd)/.venv"
  '';
}
