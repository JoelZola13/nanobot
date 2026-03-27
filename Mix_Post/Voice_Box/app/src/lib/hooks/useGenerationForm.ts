import { zodResolver } from '@hookform/resolvers/zod';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { useTranscription } from '@/lib/hooks/useTranscription';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

const generationSchema = z.object({
  text: z.string().min(1, 'Text is required').max(5000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().optional(),
  modelSize: z.enum(['1.7B', '0.6B']).optional(),
  instruct: z.string().max(500).optional(),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
}

interface StartSpeechCaptureOptions {
  selectedProfileId?: string | null;
  autoSubmitAfterTranscription?: boolean;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { toast } = useToast();
  const generation = useGeneration();
  const setAudioWithAutoPlay = usePlayerStore((state) => state.setAudioWithAutoPlay);
  const setIsGenerating = useGenerationStore((state) => state.setIsGenerating);
  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);

  useModelDownloadToast({
    modelName: downloadingModelName || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModelName,
  });

  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(generationSchema),
    defaultValues: {
      text: '',
      language: 'en',
      seed: undefined,
      modelSize: '1.7B',
      instruct: '',
      ...options.defaultValues,
    },
  });

  const transcribe = useTranscription();
  const autoSubmitProfileIdRef = useRef<string | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);

  const handleSubmit = useCallback(
    async (data: GenerationFormValues, selectedProfileId: string | null): Promise<void> => {
      if (!selectedProfileId) {
        toast({
          title: 'No profile selected',
          description: 'Please select a voice profile from the cards above.',
          variant: 'destructive',
        });
        return;
      }

      try {
        setIsGenerating(true);

        const modelName = `qwen-tts-${data.modelSize}`;
        const displayName = data.modelSize === '1.7B' ? 'Qwen TTS 1.7B' : 'Qwen TTS 0.6B';

        try {
          const modelStatus = await apiClient.getModelStatus();
          const model = modelStatus.models.find((m) => m.model_name === modelName);

          if (model && !model.downloaded) {
            setDownloadingModelName(modelName);
            setDownloadingDisplayName(displayName);
          }
        } catch (error) {
          console.error('Failed to check model status:', error);
        }

        const result = await generation.mutateAsync({
          profile_id: selectedProfileId,
          text: data.text,
          language: data.language,
          seed: data.seed,
          model_size: data.modelSize,
          instruct: data.instruct || undefined,
        });

        toast({
          title: 'Generation complete!',
          description: `Audio generated (${result.duration.toFixed(2)}s)`,
        });

        const audioUrl = apiClient.getAudioUrl(result.id);
        setAudioWithAutoPlay(audioUrl, result.id, selectedProfileId, data.text.substring(0, 50));

        form.reset();
        options.onSuccess?.(result.id);
      } catch (error) {
        toast({
          title: 'Generation failed',
          description: error instanceof Error ? error.message : 'Failed to generate audio',
          variant: 'destructive',
        });
      } finally {
        setIsGenerating(false);
        setDownloadingModelName(null);
        setDownloadingDisplayName(null);
      }
    },
    [form, generation.mutateAsync, options.onSuccess, setAudioWithAutoPlay, setIsGenerating, toast],
  );

  const submitCurrentValues = useCallback(
    async (selectedProfileId: string | null): Promise<void> => {
      const values = form.getValues();

      if (!selectedProfileId) {
        toast({
          title: 'No profile selected',
          description: 'Please select a voice profile from the cards above.',
          variant: 'destructive',
        });
        return;
      }

      if (!values.text || values.text.trim().length === 0) {
        form.setError('text', {
          type: 'manual',
          message: 'Text is required',
        });
        toast({
          title: 'Speech text required',
          description: 'Please record speech or type text to generate.',
          variant: 'destructive',
        });
        return;
      }

      const isFormValid = await form.trigger();
      if (!isFormValid) {
        return;
      }

      await handleSubmit(values, selectedProfileId);
    },
    [form, handleSubmit, toast],
  );

  const handleRecordingComplete = useCallback(
    async (blob: Blob) => {
      const autoSubmitProfileId = autoSubmitProfileIdRef.current;
      autoSubmitProfileIdRef.current = null;
      const file = new File([blob], `dictation-${Date.now()}.wav`, {
        type: blob.type || 'audio/wav',
      });

      try {
        setIsTranscribing(true);

        const currentLanguage = form.getValues('language');
        const result = await transcribe.mutateAsync({
          file,
          language: currentLanguage,
        });

        const nextText = result.text.trim();
        form.setValue('text', nextText, { shouldValidate: true });

        toast({
          title: 'Speech transcribed',
          description: nextText ? 'Text filled and ready to generate.' : 'No speech was detected.',
        });

        if (autoSubmitProfileId) {
          await submitCurrentValues(autoSubmitProfileId);
        }
      } catch (error) {
        toast({
          title: 'Transcription failed',
          description: error instanceof Error ? error.message : 'Failed to transcribe recording.',
          variant: 'destructive',
        });
      } finally {
        setIsTranscribing(false);
      }
    },
    [form, submitCurrentValues, toast, transcribe],
  );

  const {
    isRecording,
    error: recordingError,
    startRecording,
    stopRecording,
    cancelRecording,
  } = useAudioRecording({
    maxDurationSeconds: 29,
    onRecordingComplete: (blob) => {
      void handleRecordingComplete(blob);
    },
  });

  useEffect(() => {
    if (recordingError) {
      toast({
        title: 'Recording error',
        description: recordingError,
        variant: 'destructive',
      });
    }
  }, [recordingError, toast]);

  const startSpeechCapture = useCallback(
    async ({
      selectedProfileId = null,
      autoSubmitAfterTranscription = false,
    }: StartSpeechCaptureOptions = {}): Promise<void> => {
      if (autoSubmitAfterTranscription && !selectedProfileId) {
        toast({
          title: 'No profile selected',
          description: 'Please select a voice profile before recording and generating.',
          variant: 'destructive',
        });
        return;
      }

      autoSubmitProfileIdRef.current = autoSubmitAfterTranscription
        ? selectedProfileId || null
        : null;
      try {
        await startRecording();
      } catch (error) {
        autoSubmitProfileIdRef.current = null;
        toast({
          title: 'Unable to start recording',
          description: error instanceof Error ? error.message : 'Failed to start microphone.',
          variant: 'destructive',
        });
      }
    },
    [startRecording, toast],
  );

  const stopSpeechCapture = useCallback(() => {
    stopRecording();
  }, [stopRecording]);

  const cancelSpeechCapture = useCallback(() => {
    autoSubmitProfileIdRef.current = null;
    cancelRecording();
  }, [cancelRecording]);

  return {
    form,
    handleSubmit,
    isPending: generation.isPending,
    isRecording,
    isTranscribing,
    startSpeechCapture,
    stopSpeechCapture,
    cancelSpeechCapture,
  };
}
