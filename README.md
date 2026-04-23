# smart-trash

Smart Trash: сервер для загрузки, сохранения и показа изображений.

## Что нужно заранее

Для запуска в локальной сети не нужен `nginx`.

Нужно только:

1. Установить `uv`: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
2. Один раз установить зависимости:

```bash
uv sync --locked
```

3. Открыть входящий TCP-порт в Windows Firewall:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\open-firewall.ps1 -Port 8000
```

Скрипт нужно запускать из PowerShell от имени администратора.

## Запуск сервера

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\start-server.ps1
```

После этого сервер доступен в локальной сети по адресу вида `http://<IP-МАШИНЫ>:8000`.

Для режима разработки:

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\start-server.ps1 -Reload
```

Настройки можно переопределять через `.env`:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false
APP_CORS_ALLOW_ORIGINS=*
APP_MAX_UPLOAD_SIZE_BYTES=10485760
APP_UPLOADS_DIR=data/uploads
```

Проверка доступности:

```bash
curl http://<HOST>:8000/health
```

## API для внешней загрузки

Форма в браузере:

```bash
POST /images/upload
```

Multipart API:

```bash
curl -X POST "http://<HOST>:8000/api/images/upload" -F "image=@photo.jpg"
```

Raw bytes API:

```bash
curl -X POST "http://<HOST>:8000/api/images/upload/raw" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@photo.jpg"
```

При новой загрузке открытая страница обновляет изображение автоматически через WebSocket `/ws/images`.

## Замечания по эксплуатации

- `APP_CORS_ALLOW_ORIGINS=*` уже позволяет отправлять запросы из браузеров с других компьютеров и с других origin.
- Не запускайте несколько worker-процессов для текущей версии сервера: уведомления по WebSocket сейчас хранятся в памяти процесса.
