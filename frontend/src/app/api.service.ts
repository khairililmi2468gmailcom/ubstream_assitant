// frontend/src/app/api.service.ts

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://192.168.0.103:8006'; // URL API FastAPI Anda

  constructor(private http: HttpClient) { }

  /**
   * Mengirim pertanyaan ke API backend dan mengembalikan jawabannya.
   * @param query Pertanyaan pengguna.
   * @returns Sebuah Observable dengan respons API.
   */
  askQuestion(query: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/ask`, { query });
  }

  /**
   * Memeriksa status API backend.
   * @returns Sebuah Observable dengan respons status API.
   */
  checkStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/status`);
  }
}
