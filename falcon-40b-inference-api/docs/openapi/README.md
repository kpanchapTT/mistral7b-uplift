# OpenAPI

OpenAPI: https://www.openapis.org/

We are using OpenAPI v3.0.1 because it is supported by Azure APIM for import and export.

## Run locally using docker swaggerapi/swagger-ui

See: https://github.com/swagger-api/swagger-ui/blob/master/docs/usage/installation.md#docker

PWD should be `project-falcon/falcon-40b-inference-api`

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
