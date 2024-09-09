import emeraldprocessing.pipeline
import swaggerspect
import json

print(json.dumps(swaggerspect.swagger_to_json_schema(swaggerspect.get_apis("emeraldprocessing.pipeline_step")), indent=True))
