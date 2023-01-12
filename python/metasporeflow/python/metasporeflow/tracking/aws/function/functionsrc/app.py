import json

def lambda_handler(event, context):
	# print(json.dumps(event['body']))
	body = event['body']
	try:
		body_dict = json.loads(body)
	except Exception as e:
		return {
			'statusCode': 400,
			'body': json.dumps('Error. Invalid Json String in Request: ' + str(e))
		}
	if not {'user_id', 'item_id'} < set(body_dict.keys()):
		return {
			'statusCode': 400,
			'body': "Error. user_id, item_id are required in request json body"
		}
	# print log to extension
	print(body)
	return {
		'statusCode': 200,
		'body': json.dumps('Success Lambda Handler')
	}
