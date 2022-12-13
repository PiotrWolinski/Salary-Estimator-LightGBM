from app import app
from app.controler import make_prediciton

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
