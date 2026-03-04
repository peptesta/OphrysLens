from flask import Flask
from flask_cors import CORS
from app.api.inference import inference_bp
from app.model_fun.model_loader import load_resources # Resource loading function
from app.api.db_inference import db_inference_bp
from app.api.save_db import save_bp
from app.api.new_db_inference import new_db_inference_bp
from app import model_state

onevall_models = None
model = None

def create_app():
    app = Flask(__name__)
    CORS(app)

    resources = load_resources()
    model_state.load_and_set_models(resources)
    print("--- MODELS LOADED SUCCESSFULLY ---")

    app.register_blueprint(inference_bp)
    app.register_blueprint(db_inference_bp)
    app.register_blueprint(new_db_inference_bp)
    app.register_blueprint(save_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    
    # --- BLOCK FOR DEBUGGING ---
    print("\n--- REGISTERED ROUTES ---")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule}")
    print("-------------------------\n")
    # ------------------------------------

    print(f"--- SERVER LISTENING ON PORT 5000) ---")
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

