(() => {
    const storageKey = "theme";
    const root = document.documentElement;
    const themeToggle = document.querySelector("[data-theme-toggle]");
    const fileInput = document.querySelector("[data-file-input]");
    const fileName = document.querySelector("[data-file-name]");
    const imageElement = document.querySelector("[data-live-image]");
    const imageLink = document.querySelector("[data-live-link]");
    const imageFrame = document.querySelector("[data-image-frame]");
    const imageCaption = document.querySelector("[data-image-caption]");
    const emptyState = document.querySelector("[data-empty-state]");

    const applyTheme = (theme) => {
        root.dataset.theme = theme;

        if (!themeToggle) {
            return;
        }

        themeToggle.textContent = theme === "dark" ? "Светлая тема" : "Тёмная тема";
    };

    const currentTheme = root.dataset.theme || "light";
    applyTheme(currentTheme);

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
            localStorage.setItem(storageKey, nextTheme);
            applyTheme(nextTheme);
        });
    }

    if (fileInput && fileName) {
        fileInput.addEventListener("change", () => {
            const [file] = fileInput.files ?? [];
            fileName.textContent = file ? file.name : "Файл не выбран";
        });
    }

    if (!imageElement || !imageLink || !imageFrame || !imageCaption || !emptyState) {
        return;
    }

    const applyIncomingImage = (url) => {
        imageElement.src = url;
        imageLink.href = url;
        imageLink.textContent = url;
        imageFrame.classList.remove("is-hidden");
        imageCaption.classList.remove("is-hidden");
        emptyState.classList.add("is-hidden");
    };

    const connectRealtime = () => {
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        const websocket = new WebSocket(`${protocol}://${window.location.host}/ws/images`);

        websocket.addEventListener("message", (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type !== "image_uploaded" || !payload.image?.url) {
                    return;
                }

                applyIncomingImage(payload.image.url);
            } catch {
                return;
            }
        });

        websocket.addEventListener("close", () => {
            window.setTimeout(connectRealtime, 1500);
        });
    };

    connectRealtime();
})();
