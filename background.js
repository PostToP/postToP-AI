function handleUpdated(tabId, changeInfo, tabInfo) {
  if (changeInfo.status !== "complete") return;
  if (
    tabInfo.url.startsWith("https://www.youtube.com/watch?v=")
  ) {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ["content.js"],
    });
  }
}

chrome.tabs.onUpdated.addListener(handleUpdated);