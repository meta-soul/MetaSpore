# Text completion
#curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-completion -o out.completion.json

# Text to Image
#curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙"}' http://127.0.0.1:8098/api/infer/text-to-image -o out.t2i.json

# Text to Text
#curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-to-text -o out.t2t.json

# Text translation
curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-translation?model_type=zh2en -o out.translation.json
