// frontend/src/app/app.config.ts

import { ApplicationConfig, importProvidersFrom } from '@angular/core';
import { provideHttpClient } from '@angular/common/http'; // Import provideHttpClient
// provideClientHydration tidak diperlukan jika tidak menggunakan SSR
// import { provideClientHydration } from '@angular/platform-browser'; 
import { provideAnimations } from '@angular/platform-browser/animations'; // Pertahankan jika Anda menggunakan animasi

import { FormsModule } from '@angular/forms'; // Import FormsModule
import { CommonModule } from '@angular/common'; // Import CommonModule untuk ngClass, ngIf, dll.

export const appConfig: ApplicationConfig = {
  providers: [
    // provideRouter(routes), // Tidak diperlukan jika tidak ada routing
    // provideClientHydration(), // Hapus ini karena tidak ada SSR
    provideAnimations(), // Pertahankan jika Anda menggunakan animasi
    provideHttpClient(), // Menyediakan HttpClient service
    importProvidersFrom(FormsModule), // Menyediakan ngModel untuk form
    importProvidersFrom(CommonModule) // Menyediakan ngClass, ngIf, ngFor, dll.
  ]
};

