from app.app import app, socketio, read_gps_data_from_serial
import threading

if __name__ == '__main__':
    # Start GPS thread
    gps_thread = threading.Thread(target=read_gps_data_from_serial, args=('COM5', 115200))
    gps_thread.daemon = True
    gps_thread.start()

    # Run the Flask-SocketIO server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
