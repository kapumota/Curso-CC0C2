# Makefile - objetivos básicos para uso con Docker
IMAGE?=p1_rayosx_transformer:latest
CONTAINER?=p1_rayosx_transformer_ctr
PORT?=8888

.PHONY: build run stop sh jupyter clean

build:
	docker build -t $(IMAGE) .

run: stop
	# Monta el proyecto y habilita Jupyter en localhost:$(PORT)
	docker run --rm -d --name $(CONTAINER) -p $(PORT):8888 \
		-v $$(pwd):/workspace $(IMAGE)

stop:
	-@docker rm -f $(CONTAINER) 2>/dev/null || true

sh:
	docker run --rm -it -v $$(pwd):/workspace --entrypoint bash $(IMAGE)

jupyter: run
	@echo "Abrir: http://localhost:$(PORT)"

clean: stop
	@echo "(nada más que limpiar)"
