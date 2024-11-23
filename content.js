function waitforElement(selector) {
    return new Promise((resolve, reject) => {
        let counter = 0;
        const interval = setInterval(() => {
            if (counter > 20) {
                clearInterval(interval);
                reject("Element not found");
            }
            counter++;
            if (document.querySelector(selector)) {
                clearInterval(interval);
                resolve(document.querySelector(selector));
            }
        }, 200);
    });
}


async function getWatchID() {
    const { href } = document.location;
    const watchID = href.match(/v=([^&#]{5,})/)?.[1];

    const res = await fetch("http://localhost:8000/predict?ytid=" + watchID, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    const data = await res.json();
    const confidence = data["confidence"];
    const isMusic = data["is_music"];


    const ytTitleParemtDiv = await waitforElement("#above-the-fold #title h1 yt-formatted-string");
    const pill = document.getElementById("post-label");
    if (!pill) {
        const newPill = document.createElement("span");
        newPill.id = "post-label";
        newPill.style.marginRight = "10px";
        ytTitleParemtDiv.appendChild(newPill);
        pill = newPill;
    }
    pill.innerText = isMusic ? "(Music)" : "(Not Music)";
    pill.style.color = isMusic ? "green" : "red";
    const confidenceCalc = isMusic ? confidence : 100 - confidence;
    pill.title = `Confidence: ${confidenceCalc.toFixed(2)}`;

}

getWatchID();