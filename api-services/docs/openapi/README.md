# OpenAPI

OpenAPI: https://www.openapis.org/

We are using OpenAPI v3.0.1 because it is supported by Azure APIM for import and export.

## Update OpenAPI Specification

The spec is templated for the different environments (`local`, `dev`, `prod`) as defined in `generate_openapi_spec.py`:
```bash
python generate_openapi_spec.py
```

See that script for details of templating.

## Run locally using docker swaggerapi/swagger-ui

See: https://github.com/swagger-api/swagger-ui/blob/master/docs/usage/installation.md#docker

PWD should be `project-falcon/api-services`

```bash
docker pull swaggerapi/swagger-ui
docker run -p 8080:8080 -e SWAGGER_JSON=/mnt/openapi/dev/dev_tenstorrent_llm_openapi_v3.json -v $PWD/docs/openapi:/mnt/openapi swaggerapi/swagger-ui
```

Go to: http://localhost:8080

## Deploying on AWS CloudFront + S3

The `dist` folder from https://github.com/swagger-api/swagger-ui/tree/master/dist
contains the pre-built JS and HTML files to host the swagger-ui for a given OpenAPI JSON file.

The editted `swagger-initializer.js` to host `tenstorrent_llm_openapi_v3.json` are in this directory.

1. Place all the above mentioned files into an S3 bucket
2. Follow instructions in https://repost.aws/knowledge-center/cloudfront-https-requests-s3

## Development notes

Caching may exist at:
1. Browser level: delete cached data from `tenstorrent.com` to remove cache
2. CDN CloudFront: disable in caching policy, or add invalidation for `/docs/*` (this is disabled for AWS DEV CloudFront)


```bash
docker exec -it $(docker ps | grep "swaggerapi/swagger-ui" | awk '{print $1}') sh
```