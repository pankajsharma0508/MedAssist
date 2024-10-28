import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { PatientComponent } from './patient/patient.component';


export const routes: Routes = [
    { path: '', component: HomeComponent },
    { path: 'home', component: HomeComponent },
    { path: 'patient', component: PatientComponent },
  ];