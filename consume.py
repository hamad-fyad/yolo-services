import pika
import json
import boto3
from app import predict

def run_prediction(image_path,chat_id):
    
    return predict(chat_id=chat_id,img=image_path)

def callback(ch, method, properties, body):
    data = json.loads(body)
    chat_id = data['chat_id']
    img_key = data['img']
    bucket = data['bucket']

    # Run prediction
    result = run_prediction(img_key, chat_id)

    # Prepare result message
    result_message = {
        "chat_id": chat_id,
        "img": img_key,
        "prediction": result
    }

    # Send result to callback queue
    ch.basic_publish(
        exchange='',
        routing_key='predict_results',
        body=json.dumps(result_message)
    )
    print(f"Processed and sent result for chat_id={chat_id}")

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='predict', durable=True)
channel.queue_declare(queue='predict_results', durable=True)
channel.basic_consume(queue='predict', on_message_callback=callback, auto_ack=True)
print('Waiting for prediction requests...')
channel.start_consuming()