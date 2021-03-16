# AI_detectionServer_with_triton
AI Object detection server with Triton inference Server, Flask, Docker

1. Put your Object detection model in the "triton/models" directory.  
(See the link for the model directory structure Triton uses,   
link. https://github.com/triton-inference-server/server)

2. Set the ports, hosts, etc. of the flask directory, "flask/app.py".

3. Set the triton configuration in the "docker-compose.yml" file.

4. Enter the following command.  
`docker-compose up --build`  
  
  
**It's a simple example, so you can see it yourself and modify it.**  
**(Assume that you know how to use dockers, create Detection models, and set flasks)**
