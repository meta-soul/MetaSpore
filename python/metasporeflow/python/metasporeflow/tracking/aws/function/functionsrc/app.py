import json

def lambda_handler(event, context):
	print(json.dumps(event['body']))
	return {
		'statusCode': 200,
		'body': json.dumps('Success Lambda Handler')
	}
