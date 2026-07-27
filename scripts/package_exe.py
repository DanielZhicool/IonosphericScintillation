import os
import subprocess
import zipfile


def package():
    print("=== Step 1: Building PyInstaller Executable ===")
    subprocess.run(["uv", "run", "pyinstaller", "-y", "Uran4Scintillation.spec"], check=True)

    dist_dir = os.path.join("dist", "Uran4Scintillation")
    output_zip = "Uran4Scintillation-Windows-x64.zip"

    print(f"=== Step 2: Creating Zip Archive '{output_zip}' ===")
    if os.path.exists(output_zip):
        os.remove(output_zip)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(dist_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dist_dir)
                zipf.write(file_path, arcname)

    print(
        f"Successfully packaged desktop application into '{output_zip}' ({os.path.getsize(output_zip) / (1024 * 1024):.2f} MB)."
    )


if __name__ == "__main__":
    package()
