import requests

url = "http://0.0.0.0:8000/model"

res = requests.post(
    url, json={"x": ["Москва – столица России, многонациональный город на Москве-реке в западной части страны."]}
)
print(res.json())
