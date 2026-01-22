"""
AI vs Original Image Detection Module

Advanced detection using multiple deep learning and statistical signals.
Specifically tuned to catch modern AI generators (Midjourney, DALL-E, Stable Diffusion).
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
from scipy import fftpack
from scipy.stats import entropy


class AIImageDetector:
    """
    Advanced AI-generated image detector.
    
    Uses 10+ signals optimized for modern AI generators:
    - Spectral analysis (FFT patterns)
    - Local Binary Patterns (LBP) for texture
    - Benford's Law analysis on pixel values
    - Color space anomalies (unnatural gradients)
    - Edge sharpness inconsistency
    - High-frequency suppression
    - Patch-level coherence
    - Compression artifact patterns
    - EXIF metadata presence
    - Image dimensions (AI often uses specific sizes)
    """
    
    def __init__(self):
        self.threshold_ai_score = 0.50  # Lower threshold = more sensitive
        
        # Common AI generator dimensions
        self.ai_dimensions = [
            (512, 512), (768, 768), (1024, 1024),  # Stable Diffusion
            (1024, 1024), (1792, 1024), (1024, 1792),  # DALL-E 3
            (1456, 816), (1024, 1536),  # Midjourney
            (512, 768), (768, 512), (640, 640)  # Various
        ]
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image to determine if it's AI-generated.
        
        Returns:
            dict with 'is_ai_generated', 'confidence', 'signals', and 'explanation'
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Collect all detection signals
        signals = {}
        
        # 1. Image dimension check (AI generators use specific sizes)
        signals['dimension_match'] = self._check_ai_dimensions(img)
        
        # 2. Spectral analysis (FFT patterns)
        signals['spectral_anomaly'] = self._analyze_spectral_patterns(img)
        
        # 3. Local Binary Pattern texture analysis
        signals['texture_uniformity'] = self._analyze_texture_patterns(img)
        
        # 4. Benford's Law on pixel values
        signals['benfords_violation'] = self._check_benfords_law(img)
        
        # 5. Color gradient smoothness (AI over-smooths)
        signals['gradient_smoothness'] = self._analyze_color_gradients(img)
        
        # 6. High-frequency suppression
        signals['hf_suppression'] = self._analyze_frequency_domain(img)
        
        # 7. Edge sharpness consistency
        signals['edge_inconsistency'] = self._analyze_edge_consistency(img)
        
        # 8. Patch-level coherence (AI has perfect coherence)
        signals['patch_coherence'] = self._analyze_patch_coherence(img)
        
        # 9. JPEG compression patterns
        signals['compression_artifacts'] = self._analyze_compression(img)
        
        # 10. Metadata presence (weak signal)
        signals['metadata_present'] = self._check_metadata(image_path)
        
        # 11. Noise analysis (AI lacks natural sensor noise)
        signals['noise_pattern'] = self._analyze_noise_pattern(img)
        
        # Calculate AI likelihood score
        ai_score, explanation = self._calculate_ai_score(signals)
        
        return {
            'is_ai_generated': ai_score > self.threshold_ai_score,
            'ai_confidence': ai_score,
            'original_confidence': 1 - ai_score,
            'signals': signals,
            'explanation': explanation,
            'verdict': 'AI-Generated' if ai_score > self.threshold_ai_score else 'Original/Real'
        }
    
    def _check_ai_dimensions(self, img: np.ndarray) -> float:
        """Check if image dimensions match common AI generator outputs."""
        h, w = img.shape[:2]
        
        # Check for exact matches
        for ai_w, ai_h in self.ai_dimensions:
            if (w == ai_w and h == ai_h) or (w == ai_h and h == ai_w):
                return 0.9  # Strong indicator
        
        # Check if dimensions are multiples of 64 (common in AI)
        if w % 64 == 0 and h % 64 == 0:
            return 0.7
        
        # Check for power-of-2 or AI-friendly ratios
        if w % 8 == 0 and h % 8 == 0 and min(w, h) >= 512:
            return 0.5
        
        return 0.2
    
    def _analyze_spectral_patterns(self, img: np.ndarray) -> float:
        """
        Analyze FFT spectrum for AI-specific patterns.
        AI generators create characteristic spectral signatures.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Downsample for speed
        if gray.shape[0] > 512:
            gray = cv2.resize(gray, (512, 512))
        
        # 2D FFT
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Analyze radial power spectrum
        h, w = magnitude.shape
        center = (h // 2, w // 2)
        
        # Create radial bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Calculate radial average
        radial_bins = np.bincount(r.ravel(), magnitude.ravel())
        radial_count = np.bincount(r.ravel())
        radial_avg = radial_bins / (radial_count + 1e-10)
        
        # AI images have abnormal power fall-off
        # Calculate slope of log-log power spectrum
        valid_idx = radial_avg > 0
        if np.sum(valid_idx) < 10:
            return 0.5
        
        log_radial = np.log(radial_avg[valid_idx] + 1)
        log_freq = np.log(np.arange(len(radial_avg))[valid_idx] + 1)
        
        # Fit slope
        if len(log_freq) > 1:
            slope = np.polyfit(log_freq, log_radial, 1)[0]
            
            # AI images typically have slope between -2 and -3
            # Real photos have slope around -1.5
            if -3.0 < slope < -2.0:
                return 0.8
            elif -2.5 < slope < -1.8:
                return 0.6
            else:
                return 0.3
        
        return 0.5
    
    def _analyze_texture_patterns(self, img: np.ndarray) -> float:
        """
        Local Binary Patterns (LBP) analysis.
        AI images have unnaturally uniform textures.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Downsample
        if gray.shape[0] > 256:
            gray = cv2.resize(gray, (256, 256))
        
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        
        # Calculate entropy
        hist = hist[hist > 0]
        lbp_entropy = -np.sum(hist * np.log2(hist))
        
        # AI images have lower LBP entropy (more uniform)
        # Real photos: 6-8, AI: 4-6
        if lbp_entropy < 5.0:
            return 0.8
        elif lbp_entropy < 6.5:
            return 0.6
        else:
            return 0.3
    
    def _check_benfords_law(self, img: np.ndarray) -> float:
        """
        Benford's Law analysis on pixel values.
        Natural images follow Benford's Law, AI often doesn't.
        """
        # Get all pixel values
        pixels = img.reshape(-1, img.shape[-1])
        
        # Get first digits (non-zero)
        first_digits = []
        for channel in range(pixels.shape[1]):
            channel_pixels = pixels[:, channel]
            # Get first digit of non-zero values
            nonzero = channel_pixels[channel_pixels > 0]
            if len(nonzero) > 0:
                first_dig = np.array([int(str(int(p))[0]) for p in nonzero if p >= 10])
                first_digits.extend(first_dig)
        
        if len(first_digits) < 100:
            return 0.5
        
        # Calculate observed distribution
        digit_counts = np.bincount(first_digits, minlength=10)[1:10]  # digits 1-9
        digit_freq = digit_counts / digit_counts.sum()
        
        # Benford's expected distribution
        benford_expected = np.log10(1 + 1/np.arange(1, 10))
        
        # Chi-square test
        chi_square = np.sum((digit_freq - benford_expected)**2 / benford_expected)
        
        # Higher chi-square = more deviation from Benford's = more likely AI
        if chi_square > 0.15:
            return 0.8
        elif chi_square > 0.08:
            return 0.6
        else:
            return 0.3
    
    def _analyze_color_gradients(self, img: np.ndarray) -> float:
        """
        Analyze color gradient smoothness.
        AI over-smooths gradients.
        """
        # Convert to LAB for perceptual gradients
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
        
        # Calculate gradients
        grad_l_x = cv2.Sobel(lab[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        grad_l_y = cv2.Sobel(lab[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_l_x**2 + grad_l_y**2)
        
        # Calculate gradient histogram
        hist, bins = np.histogram(grad_magnitude.flatten(), bins=50, range=(0, 100))
        hist = hist.astype(float) / hist.sum()
        
        # AI images have very peaked distribution (overly smooth)
        peak_value = hist.max()
        
        if peak_value > 0.3:
            return 0.8  # Very peaked = AI
        elif peak_value > 0.2:
            return 0.6
        else:
            return 0.3
    
    def _analyze_patch_coherence(self, img: np.ndarray) -> float:
        """
        Analyze patch-level coherence.
        AI images have suspiciously perfect coherence across patches.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Divide into patches
        patch_size = 32
        h, w = gray.shape
        
        patch_stats = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patch_stats.append({
                    'mean': np.mean(patch),
                    'std': np.std(patch)
                })
        
        if len(patch_stats) < 4:
            return 0.5
        
        # Calculate variance of patch statistics
        mean_var = np.var([p['mean'] for p in patch_stats])
        std_var = np.var([p['std'] for p in patch_stats])
        
        # AI images have suspiciously low variance across patches
        coherence_score = 1.0 / (1.0 + mean_var / 1000.0)
        
        if coherence_score > 0.8:
            return 0.8
        elif coherence_score > 0.6:
            return 0.6
        else:
            return 0.3
    
    def _analyze_frequency_domain(self, img: np.ndarray) -> float:
        """
        High-frequency suppression analysis.
        AI generators suppress high frequencies (over-smoothing).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Analyze high-frequency content
        rows, cols = magnitude.shape
        center_row, center_col = rows // 2, cols // 2
        
        # Define frequency bands
        low_band = magnitude[
            int(center_row * 0.7):int(center_row * 1.3),
            int(center_col * 0.7):int(center_col * 1.3)
        ]
        high_band_mask = np.ones_like(magnitude)
        high_band_mask[
            int(center_row * 0.7):int(center_row * 1.3),
            int(center_col * 0.7):int(center_col * 1.3)
        ] = 0
        high_band = magnitude * high_band_mask
        
        # Calculate energy ratio
        low_energy = np.sum(low_band ** 2)
        high_energy = np.sum(high_band ** 2)
        hf_ratio = high_energy / (low_energy + high_energy + 1e-10)
        
        # AI images have suspiciously low high-frequency content
        if hf_ratio < 0.05:
            return 0.9  # Very low = definitely AI
        elif hf_ratio < 0.15:
            return 0.7
        elif hf_ratio < 0.25:
            return 0.5
        else:
            return 0.2
    
    def _analyze_noise_pattern(self, img: np.ndarray) -> float:
        """
        Deep noise analysis. AI lacks natural sensor noise.
        Real photos have characteristic noise from camera sensors.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        
        # Multi-scale noise analysis
        noise_scores = []
        
        for sigma in [1, 2, 3]:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            noise = gray - blurred
            
            # Analyze noise statistics
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # Real photos: noise_std typically 2-10
            # AI images: noise_std typically < 1 or artificially added > 15
            if noise_std < 1.0:
                noise_scores.append(0.9)  # Too clean = AI
            elif 1.0 <= noise_std < 2.5:
                noise_scores.append(0.7)
            elif 2.5 <= noise_std < 12:
                noise_scores.append(0.2)  # Normal range
            else:
                noise_scores.append(0.5)  # Artificial noise added
        
        return np.mean(noise_scores)
    
    def _analyze_edge_consistency(self, img: np.ndarray) -> float:
        """
        Analyze edge sharpness consistency.
        AI images have unnaturally consistent edge sharpness.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density
        fine_density = np.sum(edges_fine > 0) / edges_fine.size
        coarse_density = np.sum(edges_coarse > 0) / edges_coarse.size
        
        # Calculate edge sharpness variance
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Get edge pixels
        edge_pixels = edge_magnitude[edges_fine > 0]
        if len(edge_pixels) < 100:
            return 0.5
        
        edge_variance = np.var(edge_pixels)
        
        # AI images have suspiciously low edge variance (too consistent)
        if edge_variance < 500:
            return 0.8
        elif edge_variance < 1500:
            return 0.6
        else:
            return 0.3
    
    def _check_metadata(self, image_path: str) -> bool:
        """
        Check for EXIF metadata. Real photos usually have camera metadata,
        AI-generated images typically don't.
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if exif_data is None:
                return False
            
            # Check for camera-specific tags
            camera_tags = ['Make', 'Model', 'DateTime', 'FNumber', 'ExposureTime']
            found_camera_tags = 0
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in camera_tags:
                    found_camera_tags += 1
            
            return found_camera_tags >= 2
        except:
            return False
    
    def _analyze_compression(self, img: np.ndarray) -> float:
        """
        Analyze compression artifacts.
        Real JPEGs have characteristic block artifacts,
        AI images may have different compression patterns.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Look for 8x8 DCT block boundaries (JPEG artifacts)
        h, w = gray.shape
        block_diffs = []
        
        for i in range(8, h - 8, 8):
            for j in range(8, w - 8, 8):
                # Check vertical boundary
                diff_v = abs(int(gray[i, j]) - int(gray[i-1, j]))
                # Check horizontal boundary
                diff_h = abs(int(gray[i, j]) - int(gray[i, j-1]))
                block_diffs.append(diff_v + diff_h)
        
        if not block_diffs:
            return 0.5
        
        avg_block_diff = np.mean(block_diffs)
        
        # Real JPEGs typically have visible block artifacts
        if avg_block_diff < 2:
            return 0.6  # Too smooth (possibly AI)
        else:
            return 0.3  # Normal JPEG artifacts
    
    def _calculate_ai_score(self, signals: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate overall AI likelihood score from all signals.
        Uses weighted combination optimized for modern AI generators.
        
        Returns:
            (score, explanation) where score is 0-1 (1 = definitely AI)
        """
        # Optimized weights based on signal reliability
        weights = {
            'dimension_match': 0.12,        # Strong indicator for AI
            'spectral_anomaly': 0.12,       # Spectral patterns are reliable
            'texture_uniformity': 0.10,     # LBP texture analysis
            'benfords_violation': 0.08,     # Benford's Law
            'gradient_smoothness': 0.10,    # Over-smoothing
            'hf_suppression': 0.15,         # High-freq suppression (strongest)
            'edge_inconsistency': 0.08,     # Edge patterns
            'patch_coherence': 0.10,        # Patch-level analysis
            'compression_artifacts': 0.05,  # Compression patterns
            'metadata_present': 0.05,       # EXIF (weakest, easily faked)
            'noise_pattern': 0.05,          # Noise analysis
        }
        
        score = 0.0
        explanations = []
        
        # Dimension check
        if signals['dimension_match'] > 0.7:
            score += signals['dimension_match'] * weights['dimension_match']
            explanations.append("Matches common AI generator dimensions")
        else:
            score += signals['dimension_match'] * weights['dimension_match']
        
        # Spectral anomaly
        if signals['spectral_anomaly'] > 0.6:
            score += signals['spectral_anomaly'] * weights['spectral_anomaly']
            explanations.append("Abnormal FFT spectral signature")
        else:
            score += signals['spectral_anomaly'] * weights['spectral_anomaly']
        
        # Texture uniformity
        if signals['texture_uniformity'] > 0.6:
            score += signals['texture_uniformity'] * weights['texture_uniformity']
            explanations.append("Unnaturally uniform texture patterns (LBP)")
        else:
            score += signals['texture_uniformity'] * weights['texture_uniformity']
        
        # Benford's Law violation
        if signals['benfords_violation'] > 0.6:
            score += signals['benfords_violation'] * weights['benfords_violation']
            explanations.append("Violates Benford's Law distribution")
        else:
            score += signals['benfords_violation'] * weights['benfords_violation']
        
        # Gradient smoothness
        if signals['gradient_smoothness'] > 0.6:
            score += signals['gradient_smoothness'] * weights['gradient_smoothness']
            explanations.append("Over-smoothed color gradients")
        else:
            score += signals['gradient_smoothness'] * weights['gradient_smoothness']
        
        # High-frequency suppression (strongest signal)
        if signals['hf_suppression'] > 0.7:
            score += signals['hf_suppression'] * weights['hf_suppression']
            explanations.append("Severe high-frequency suppression (AI smoothing)")
        else:
            score += signals['hf_suppression'] * weights['hf_suppression']
        
        # Edge inconsistency
        if signals['edge_inconsistency'] > 0.6:
            score += signals['edge_inconsistency'] * weights['edge_inconsistency']
            explanations.append("Inconsistent edge sharpness")
        else:
            score += signals['edge_inconsistency'] * weights['edge_inconsistency']
        
        # Patch coherence
        if signals['patch_coherence'] > 0.6:
            score += signals['patch_coherence'] * weights['patch_coherence']
            explanations.append("Suspiciously perfect patch-level coherence")
        else:
            score += signals['patch_coherence'] * weights['patch_coherence']
        
        # Compression artifacts
        score += signals['compression_artifacts'] * weights['compression_artifacts']
        if signals['compression_artifacts'] > 0.6:
            explanations.append("Unusual compression artifact patterns")
        
        # Metadata (weak signal)
        metadata_score = 0.7 if not signals['metadata_present'] else 0.2
        score += metadata_score * weights['metadata_present']
        if not signals['metadata_present']:
            explanations.append("Missing camera EXIF metadata")
        
        # Noise pattern
        if signals['noise_pattern'] > 0.7:
            score += signals['noise_pattern'] * weights['noise_pattern']
            explanations.append("Lacks natural sensor noise")
        else:
            score += signals['noise_pattern'] * weights['noise_pattern']
        
        if not explanations:
            explanations.append("Image appears to be a genuine photograph")
        elif len(explanations) >= 4:
            explanations.insert(0, f"Multiple AI indicators detected ({len(explanations)} signals)")
        
        explanation = "; ".join(explanations)
        
        return score, explanation


def detect_ai_image(image_path: str) -> Dict[str, Any]:
    """Convenience function to detect AI-generated images."""
    detector = AIImageDetector()
    return detector.analyze(image_path)
