// 確保所有 MediaPipe 模組已導入
import {
    ImageClassifier,
    FilesetResolver, 
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

// ⭐ DOMContentLoaded 監聽器
document.addEventListener('DOMContentLoaded', () => {

    // --- DOM 元素引用 ---
    const demosSection = document.getElementById("demos"); 
    const modelStatus = document.getElementById("modelStatus");
    const fileInput = document.getElementById('fileInput');
    const uploadedImage = document.getElementById('uploadedImage');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    const uploadResultDisplay = document.getElementById('uploadResultDisplay');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    let video = document.getElementById("webcam");
    
    // ⭐ 更新 DOM 引用以匹配 HTML 中的新 ID
    let enableWebcamButton = document.getElementById("enableWebcamButton");
    let disableWebcamButton = document.getElementById("disableWebcamButton"); // 新增關閉按鈕引用
    
    let classificationDisplayElement = document.getElementById("classificationResultDisplay");
    const modelSelector = document.getElementById("modelSelector");
    const modelStatusDetail = document.getElementById("modelStatusDetail");


    // --- 狀態變數與設定 ---
    let imageClassifierForImage;  
    let imageClassifierForVideo;  
    
    // ⭐ 新增：用於儲存視訊串流的變數
    let videoStream = null; 

    // 靜態模型清單
    const MODEL_LIST = [
        { name: 'X-Ray (肺炎)', path: 'models/X-Ray_0.2.tflite' },
        { name: 'Flower_tw6 (花卉)', path: 'models/Flower_tw6.tflite' }, 
        { name: 'EfficientNet-Lite0 (官方雲端模型)', path: 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite' },
    ];
    
    // 中文翻譯對應表
    const categoryTranslations = {
        "normal": "正常", 
        "pneumonia": "肺炎", 
        "tuberculosis": "結核病",
    }; 

    // 初始化模型選擇器 (使用靜態清單)
    const initializeModelSelector = () => {
        modelSelector.innerHTML = ''; 
        MODEL_LIST.forEach((model, index) => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            if (index === 0) {
                option.selected = true;
            }
            modelSelector.appendChild(option);
        });
        if (MODEL_LIST.length > 0) {
             modelStatusDetail.textContent = `當前模型: ${MODEL_LIST[0].name} (${MODEL_LIST[0].path.split('/').pop()})`;
        }
    };


    /********************************************************************
    // 核心函數：雙分類器初始化
    ********************************************************************/

    const initializeImageClassifier = async () => {
        const selectedModelPath = modelSelector.value;
        if (!selectedModelPath) {
             modelStatus.textContent = "❌ 請先選擇一個模型。";
             return;
        }
        
        // 確保關閉鏡頭，防止切換模型時資源衝突
        disableCam(); 
        
        modelStatus.textContent = "正在載入模型...";
        demosSection.classList.add("invisible"); 
        
        // 釋放舊資源
        if (imageClassifierForImage) imageClassifierForImage.close();
        if (imageClassifierForVideo) imageClassifierForVideo.close();
        
        // 禁用按鈕防止在載入中操作
        if (enableWebcamButton) enableWebcamButton.disabled = true;
        if (fileInput) fileInput.disabled = true;

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
        );
        
        try {
            const baseOptions = {
                modelAssetPath: selectedModelPath,
                // delegate: "GPU" 
            };
            const commonOptions = {
                baseOptions: baseOptions,
                scoreThreshold: 0.1, 
                maxResults: 5 
            };

            // 1. 初始化 IMAGE 模式分類器
            imageClassifierForImage = await ImageClassifier.createFromOptions(vision, {
                ...commonOptions,
                runningMode: "IMAGE",
            });

            // 2. 初始化 VIDEO 模式分類器
            imageClassifierForVideo = await ImageClassifier.createFromOptions(vision, {
                ...commonOptions,
                runningMode: "VIDEO",
            });

            demosSection.classList.remove("invisible");
            modelStatus.textContent = "✅ 模型載入完成。";
            if (enableWebcamButton) enableWebcamButton.disabled = false;
            if (fileInput) fileInput.disabled = false;
            
            // 更新狀態細節
            const selectedModelName = modelSelector.options[modelSelector.selectedIndex].text;
            modelStatusDetail.textContent = `當前模型: (${selectedModelPath.split('/').pop()})`;

            // 清除之前的分類結果
            uploadResultDisplay.textContent = "請上傳圖片";
            uploadedImageContainer.querySelector(".info")?.remove();


        } catch (e) {
            modelStatus.textContent = `❌ 模型載入失敗: ${e.message}。請檢查模型路徑。`;
            console.error("模型載入失敗:", e);
            if (enableWebcamButton) enableWebcamButton.disabled = true;
            if (fileInput) fileInput.disabled = true;
        }
    };
    
    // 載入清單並首次初始化
    initializeModelSelector();
    modelSelector.addEventListener('change', initializeImageClassifier);
    initializeImageClassifier(); 


    /********************************************************************
    // 核心分類函數：專門用於靜態圖片
    ********************************************************************/
    
    async function classifyStaticImage(imgElement) {
        if (!imageClassifierForImage) {
            uploadResultDisplay.textContent = "圖片分類器尚未載入完成，請稍候。";
            return null;
        }
        try {
            const classificationResult = imageClassifierForImage.classify(imgElement);
            return classificationResult;
        } catch (error) {
            console.error("靜態圖片分類失敗:", error);
            uploadResultDisplay.textContent = `❌ 分類失敗: ${error.message}`;
            return null;
        }
    }


    function displayStaticImageClassification(result) {
        const resultContainer = uploadedImageContainer;
        
        const existingP = resultContainer.querySelector(".info");
        if (existingP) existingP.remove();

        if (!result || !result.classifications || result.classifications.length === 0) {
            uploadResultDisplay.textContent = "未檢測到分類結果。";
            return;
        }

        const categories = result.classifications[0]?.categories;

        if (!categories || categories.length === 0) {
            uploadResultDisplay.textContent = "未檢測到有效分類。";
            return;
        }
        
        const p = document.createElement("p");
        p.setAttribute("class", "info");

        const SCORE_THRESHOLD = imageClassifierForImage.options.scoreThreshold || 0.1;
        
        const filteredCategories = categories.filter(c => 
            parseFloat(c.score) >= SCORE_THRESHOLD
        );

        if (filteredCategories.length === 0) {
            p.innerText = `未檢測到高於 ${Math.round(SCORE_THRESHOLD * 100)}% 準確率的分類結果。`;
        } else {
            p.innerText = filteredCategories.map(c => {
                const translatedName = categoryTranslations[c.categoryName.toLowerCase()] || c.categoryName;
                return `${translatedName}: ${Math.round(parseFloat(c.score) * 100)}%`;
            }).join(' | ');
        }
        
        resultContainer.appendChild(p);
    }


    /********************************************************************
    // 上傳圖片分類事件處理
    ********************************************************************/
    
    function processImageFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                uploadResultDisplay.textContent = "請選擇或貼上有效的圖片檔案。";
                // 清除檔案名顯示
                if(fileNameDisplay) fileNameDisplay.textContent = "";
                return;
            }

            const oldInfo = uploadedImageContainer.querySelector(".info");
            if (oldInfo) oldInfo.remove();
            
            // 清空 input file
            fileInput.value = '';

            // ⭐ 顯示檔案名稱
            if (fileNameDisplay) {
                fileNameDisplay.textContent = `${file.name}`;
            }

            uploadResultDisplay.textContent = "圖片載入中...";
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block'; 
                uploadedImage.onload = async () => {
                    await handleUploadClassification();
                }
            };
            reader.readAsDataURL(file);
        }
        
        // 另外，在文件載入完成 (DOMContentLoaded) 時，確保初始化這個顯示區域
        if (fileNameDisplay) {
            fileNameDisplay.textContent = "";
        }

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            processImageFile(file);
        } else {
            uploadedImage.style.display = 'none';
            const oldInfo = uploadedImageContainer.querySelector(".info");
            if (oldInfo) oldInfo.remove();
            uploadResultDisplay.textContent = "請選擇一張圖片進行分類。";
        }
    });

    uploadedImageContainer.addEventListener('paste', (event) => {
        event.preventDefault(); 
        const items = (event.clipboardData || event.originalEvent.clipboardData).items;
        let imageFile = null;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                imageFile = items[i].getAsFile();
                break;
            }
        }

        if (imageFile) {
            processImageFile(imageFile);
            fileInput.value = '';
        } else {
            uploadResultDisplay.textContent = "未從剪貼簿中找到圖片。";
        }
    });

    uploadedImage.addEventListener('click', handleUploadClassification);

    async function handleUploadClassification() {
        if (!imageClassifierForImage) {
            uploadResultDisplay.textContent = "圖片分類器尚未載入，請稍候。";
            return;
        }

        uploadResultDisplay.textContent = "正在判定中...";
        
        if (uploadedImage.src.length === 0 || uploadedImage.style.display === 'none') {
            uploadResultDisplay.textContent = "❌ 請先選擇或貼上一張圖片。";
            return;
        }

        try {
            const classificationResult = await classifyStaticImage(uploadedImage);

            if (classificationResult) {
                displayStaticImageClassification(classificationResult);
                uploadResultDisplay.textContent = "✅ 判定完成。點擊圖片可再次判定。";
            }
        } catch (e) {
            console.error("上傳圖片分類處理失敗:", e);
            uploadResultDisplay.textContent = `❌ 判定失敗： ${e.message}`;
        }
    }


    /********************************************************************
    // 視訊鏡頭分類
    ********************************************************************/

    if (classificationDisplayElement) {
        classificationDisplayElement.textContent = "等待啟用視訊鏡頭...";
    }

    function hasGetUserMedia() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    // ⭐ 關閉視訊鏡頭函數
    function disableCam() {
        if (videoStream) {
            // 停止所有軌道 (Tracks)，釋放鏡頭
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
        
        // 移除 video 元素的 srcObject
        video.srcObject = null;
        // 確保 predictWebcam 迴圈停止 (雖然 stream 停止會隱式停止 MediaPipe task，但這是 UI 上的保險)

        // 切換按鈕和 UI 狀態
        enableWebcamButton.classList.remove("removed");
        disableWebcamButton.classList.add("removed");
        classificationDisplayElement.textContent = "等待啟用視訊鏡頭...";

        console.log("視訊鏡頭已關閉。");
    }


    if (hasGetUserMedia()) {
        if (enableWebcamButton) {
            enableWebcamButton.addEventListener("click", enableCam);
        }
        // ⭐ 新增：關閉按鈕的事件監聽器
        if (disableWebcamButton) {
             disableWebcamButton.addEventListener("click", disableCam);
        }
    } else {
        console.warn("getUserMedia() is not supported by your browser");
        if (enableWebcamButton) {
            enableWebcamButton.textContent = "瀏覽器不支持視訊鏡頭";
            enableWebcamButton.disabled = true;
        }
    }

    async function enableCam(event) {
        if (!imageClassifierForVideo) {
            alert("視訊分類器尚未載入完成，請稍候。");
            return;
        }

        // 隱藏啟用按鈕，顯示關閉按鈕
        enableWebcamButton.classList.add("removed");
        disableWebcamButton.classList.remove("removed");

        const constraints = {
            video: true
        };

        // Activate the webcam stream.
        navigator.mediaDevices
            .getUserMedia(constraints)
            .then(function (stream) {
                // ⭐ 儲存串流對象
                videoStream = stream;
                video.srcObject = stream;
                // 確保 video 元素載入完資料再開始預測
                video.addEventListener("loadeddata", predictWebcam);
            })
            .catch((err) => {
                console.error("無法取得視訊串流:", err);
                // 發生錯誤時，將 UI 狀態恢復
                enableWebcamButton.classList.remove("removed");
                disableWebcamButton.classList.add("removed");
                alert("無法取得視訊鏡頭串流。請檢查權限或您的裝置。");
            });
    }

    let lastVideoTime = -1;
    function predictWebcam() {
        // ⭐ 檢查串流是否仍然存在
        if (!imageClassifierForVideo || !videoStream) {
            return; 
        }
        
        let startTimeMs = performance.now(); 

        if (video.currentTime !== lastVideoTime) { 
            lastVideoTime = video.currentTime;
            const classificationResult = imageClassifierForVideo.classifyForVideo(video, startTimeMs);
            displayVideoClassification(classificationResult);
        }
        
        window.requestAnimationFrame(predictWebcam);
    }


    function displayVideoClassification(result) {
        if (!classificationDisplayElement || !imageClassifierForVideo) return;

        const SCORE_THRESHOLD = imageClassifierForVideo.options.scoreThreshold || 0.1; 

        const categories = result.classifications[0]?.categories;
        
        if (!categories || categories.length === 0) {
            classificationDisplayElement.innerText = "未檢測到有效分類...";
            return;
        }

        const filteredCategories = categories.filter(c => 
            parseFloat(c.score) >= SCORE_THRESHOLD
        );

        if (filteredCategories.length === 0) {
            classificationDisplayElement.innerText = `未檢測到高於 ${Math.round(SCORE_THRESHOLD * 100)}% 準確率的分類結果。`;
            return;
        }

        classificationDisplayElement.innerText = "分類結果: " + filteredCategories.map(c => {
            const translatedName = categoryTranslations[c.categoryName.toLowerCase()] || c.categoryName;
            return `${translatedName} ${Math.round(parseFloat(c.score) * 100)}%`;
        }).join(' | ');
    }
}); // <-- DOMContentLoaded 結束