<?php

use App\Models\User;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('welcome');
});

Route::get('/login', function () {
    $user = User::firstOrCreate(
        ['email' => 'admin@mixpost.local'],
        ['name' => 'Mixpost Demo', 'password' => Hash::make('password')]
    );

    Auth::login($user);

    return redirect()->intended('/mixpost');
})->name('login');
