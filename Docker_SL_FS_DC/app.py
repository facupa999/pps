from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/segmentar', methods=['POST'])
def segmentar():
    img_path = request.json.get('img_path')  # Ruta a la imagen T1
    output_dir = "/shared"  # Directorio compartido donde se almacenará el resultado
    name_subject = request.json.get('name_subject')  # Nombre del sujeto para identificar la salida

    if not img_path or not name_subject:
        return jsonify({"error": "Missing img_path or name_subject"}), 400

    try:
        # Ejecuta el script de segmentación con los parámetros adecuados
        command = ["/app/run_pipeline.sh", img_path, output_dir, name_subject]
        subprocess.run(command, check=True)

        # Definir la ruta final del archivo de salida
        final_path = os.path.join(output_dir, f"{name_subject}.mgz")
        
        # Verifica que el archivo se haya generado
        if os.path.exists(final_path):
            return jsonify({"message": "Segmentación completada exitosamente.", "output_path": final_path}), 200
        else:
            return jsonify({"error": "Error: El archivo de salida no se generó correctamente."}), 500

    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
