# Locust load testing

## install

```bash
# requires dev dependencies
pip install -r requirements-dev.txt
```

## setup

edit `locust_load_test.py`:

```
TEST_PROMPTS = [ ... ]
...
self.model_name = "predictions/falcon40b"
self.api_key = f"{os.environ['AUTHORIZATION']}"
```

edit `locust.conf`:

```
host = <API_URL>
```

## run

```bash
python -m locust
```

Output looks like:
```
Response time percentiles (approximated)
Type     Name                                   50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|---------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
POST     /predictions/named_entity_recognition/       360    400    430    450    480    510    530    540    560    580    580   2171
--------|---------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                             360    400    430    450    480    510    530    540    560    580    580   2171
```