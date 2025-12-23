/**
 * Custom hook for voice recording and transcription
 * Extracts voice recording logic from Chat component
 */

import { useState, useRef } from 'react';
import { chatAPI } from '../utils/api';

export function useVoiceRecording(onTranscriptionComplete) {
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const abortControllerRef = useRef(null);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                setIsTranscribing(true);
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

                abortControllerRef.current = new AbortController();

                try {
                    const res = await chatAPI.transcribe(audioBlob);
                    if (res.data.text) {
                        onTranscriptionComplete?.(res.data.text);
                    }
                } catch (error) {
                    if (error.name !== 'CanceledError' && error.message !== 'canceled') {
                        console.error('Transcription error:', error);
                        alert('Failed to transcribe audio. Please try again.');
                    }
                } finally {
                    setIsTranscribing(false);
                    abortControllerRef.current = null;
                }

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Microphone access error:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const cancelTranscription = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            setIsTranscribing(false);
        }
    };

    return {
        isRecording,
        isTranscribing,
        startRecording,
        stopRecording,
        cancelTranscription,
    };
}

export default useVoiceRecording;
