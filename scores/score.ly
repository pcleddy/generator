\version "2.24.0"
\header {
  title = "Ambient Drift with Cage Interruptions"
  subtitle = "Algorithmic composition — seed 99"
  composer = "CloudAutomat Labs / generator"
}

\paper {
  #(set-paper-size "a3" 'landscape)
}

% NOTE: This is a simplified reduction.
% The full graphical score (score_graphical.png) is the
% authoritative notation for this piece.

\markup {
  \column {
    \line { "INTERRUPTION TIMELINE:" }
    \vspace #1
    \line { "31.0s: cascade [mixed] — C4, D3, E4, A4, G#5 (+6 more)" }
    \line { "47.5s: cascade [mixed] — C5, C#4, E5, F#2, C2 (+7 more)" }
    \line { "70.0s: cluster [screw] — A#4, G#3, C4, D#3, C4 (+5 more)" }
    \line { "83.0s: cascade [mixed] — C2, C#3, D5, D#3, C5 (+2 more)" }
    \line { "115.5s: cascade [mixed] — C3, D3, F#4, D#3, G#3 (+7 more)" }
  }
}