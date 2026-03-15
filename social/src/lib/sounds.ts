"use client";

let messageAudio: HTMLAudioElement | null = null;
let callAudio: HTMLAudioElement | null = null;

function getMessageSound(): HTMLAudioElement {
  if (!messageAudio) {
    messageAudio = new Audio("/sounds/message.mp3");
    messageAudio.volume = 0.3;
  }
  return messageAudio;
}

function getCallSound(): HTMLAudioElement {
  if (!callAudio) {
    callAudio = new Audio("/sounds/call.mp3");
    callAudio.volume = 0.5;
    callAudio.loop = true;
  }
  return callAudio;
}

export function playMessageSound(): void {
  try {
    const audio = getMessageSound();
    audio.currentTime = 0;
    audio.play().catch(() => {}); // Ignore autoplay policy errors
  } catch {
    // Audio not available
  }
}

export function playCallRingtone(): void {
  try {
    getCallSound().play().catch(() => {});
  } catch {}
}

export function stopCallRingtone(): void {
  try {
    const audio = getCallSound();
    audio.pause();
    audio.currentTime = 0;
  } catch {}
}
