from flask import Flask, request, jsonify
import subprocess
import os
import threading
#from nppy import NPP 

app = Flask(__name__)


def ejecutar_segmentacion(img_path, output_dir, name_subject):
    try:
        command = ["/app/run_pipeline.sh", img_path, output_dir, name_subject]
        subprocess.run(command, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error en la segmentación: {str(e)}") 

@app.route('/segmentar', methods=['POST'])
def segmentar():
    img_path = request.json.get('img_path')  # Ruta a la imagen T1
    output_dir = "/shared"  # Directorio compartido donde se almacenará el resultado
    name_subject = request.json.get('name_subject')  # Nombre del sujeto

    if not img_path or not name_subject:
        return jsonify({"error": "Missing img_path or name_subject"}), 400

    try:
        # Iniciar el proceso de segmentación en un hilo separado
        threading.Thread(target=ejecutar_segmentacion, args=(img_path, output_dir, name_subject)).start()

        # Responder inmediatamente que la segmentación ha sido iniciada
        return jsonify({"message": "Segmentación iniciada exitosamente."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
