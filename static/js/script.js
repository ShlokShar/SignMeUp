const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-upload");

["dragenter", "dragover"].forEach(eventName => {
  dropArea.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropArea.classList.add("border-blue-500", "bg-blue-50");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropArea.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropArea.classList.remove("border-blue-500", "bg-blue-50");
  });
});

dropArea.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  fileInput.files = files;
  uploadFile(files[0]);
});

fileInput.addEventListener("change", (e) => {
  uploadFile(e.target.files[0]);
});

function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    console.log("Server response:", data);
  })
  .catch(err => {
    console.error("Upload error:", err);
  });
}