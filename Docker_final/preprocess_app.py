from flask import Flask, request, jsonify
import os
import subprocess
import shutil
import uuid

app = Flask(__name__)

WORK_DIR = "/tmp/nppy_processing"  # Carpeta de trabajo temporal
OUTPUT_DIR = "/tmp/nppy_output"  # Carpeta de salida

# Crear carpetas si no existen
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ejecutar_nppy(input_file):
    """Ejecuta el comando nppy para procesar una imagen aislada"""
    try:
        if not os.path.isfile(input_file):
            raise Exception(f"❌ El archivo de entrada no existe: {input_file}")
        
        # Crear un directorio único para la imagen dentro de WORK_DIR
        unique_id = str(uuid.uuid4())
        isolated_input_dir = os.path.join(WORK_DIR, unique_id)
        isolated_output_dir = os.path.join(OUTPUT_DIR, unique_id)
        os.makedirs(isolated_input_dir, exist_ok=True)
        os.makedirs(isolated_output_dir, exist_ok=True)
        
        # Mover el archivo al directorio aislado
        filename = os.path.basename(input_file)
        isolated_input_path = os.path.join(isolated_input_dir, filename)
        shutil.copy(input_file, isolated_input_path)
        
        # Ejecutar nppy
        command = f"/usr/local/bin/nppy -i {isolated_input_path} -o {isolated_output_dir}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = f"❌ Error en el procesamiento: {result.stderr.strip()}"
            print(error_msg)
            return False, error_msg
        
        # Obtener los archivos generados en la carpeta de salida
        processed_files = os.listdir(isolated_output_dir)
        
        # Buscar el archivo con terminación _norm.nii
        norm_file = next((f for f in processed_files if f.endswith("_norm.nii")), None)
        if norm_file:
            norm_file_path = os.path.join(isolated_output_dir, norm_file)
            shutil.copy(norm_file_path, input_file)  # Sobrescribir manteniendo el nombre original
        
        # Eliminar directorios temporales
        shutil.rmtree(isolated_input_dir, ignore_errors=True)
        shutil.rmtree(isolated_output_dir, ignore_errors=True)
        
        return True, {
            "message": "Procesamiento completado exitosamente",
            "processed_files": processed_files
        }
    except Exception as e:
        error_msg = f"🔥 Error en el preprocesamiento: {str(e)}"
        return False, error_msg

@app.route('/preprocess', methods=['POST'])
def preprocess():
    input_file = request.json.get('img_path')
    
    if not input_file:
        return jsonify({"error": "Missing input_file"}), 400
    
    success, message = ejecutar_nppy(input_file)
    if success:
        return jsonify({
            "success": True,
            "message": message["message"],
            "processed_files": message["processed_files"]
        }), 200
    else:
        return jsonify({
            "success": False,
            "error": message
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
